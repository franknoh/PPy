/* The PPY native async runtime (spec 77): one loop per process, epoll on Linux.

   A frame is an array of words a compiled coroutine keeps its state in; the
   runtime owns its first three (its future, the future it awaits, its state)
   and resumes it through the function `spawn` was given. A future is done or
   failed with sixty-four bits of value, and wakes the frames waiting on it.
   Timers are a small heap; sockets are non-blocking and waited on through
   epoll, one pending operation per direction. Nothing here blocks except the
   wait itself, and nothing is faked: an operation that cannot complete
   completes with a negative errno. */

#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1
#endif

#include "ppy_aio.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#define PPY_AIO_EPOLL 1
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#endif

enum { PENDING = 0, DONE = 1, FAILED = 2 };
enum { SLOT_SELF = 0, SLOT_AWAITING = 1, SLOT_STATE = 2 };

typedef struct future {
    int32_t state;
    int32_t consumed;
    int32_t started;
    int64_t *frame;
    int64_t bits;
    int64_t **waiters;
    int64_t waiter_count;
    int64_t waiter_room;
} future;

typedef struct frame_entry {
    int64_t *frame;
    ppy_aio_resume_fn resume;
} frame_entry;

typedef struct timer {
    double deadline;
    future *result;
} timer;

enum { OP_NONE = 0, OP_ACCEPT, OP_CONNECT, OP_READ, OP_WRITE };

typedef struct pending_io {
    int kind;
    future *result;
    uint8_t *buffer;
    int64_t count;
    int64_t done;
} pending_io;

static frame_entry *ready = NULL;
static int64_t ready_count = 0, ready_room = 0;
static timer *timers = NULL;
static int64_t timer_count = 0, timer_room = 0;
static pending_io **pending = NULL; /* by descriptor */
static int64_t pending_room = 0;
static int64_t pending_count = 0;
#if PPY_AIO_EPOLL
static int poller = -1;
#endif

/* -- frames and their resume functions ------------------------------------------ */

typedef struct registered {
    int64_t *frame;
    ppy_aio_resume_fn resume;
} registered;

static registered *frames = NULL;
static int64_t frame_count = 0, frame_room = 0;

static ppy_aio_resume_fn resume_of(int64_t *frame) {
    for (int64_t i = 0; i < frame_count; i++) {
        if (frames[i].frame == frame) return frames[i].resume;
    }
    return NULL;
}

static void forget_frame(int64_t *frame) {
    for (int64_t i = 0; i < frame_count; i++) {
        if (frames[i].frame == frame) {
            frames[i] = frames[--frame_count];
            return;
        }
    }
}

static void schedule(int64_t *frame) {
    if (ready_count == ready_room) {
        ready_room = ready_room ? ready_room * 2 : 16;
        ready = (frame_entry *)realloc(ready, (size_t)ready_room * sizeof(frame_entry));
    }
    ready[ready_count].frame = frame;
    ready[ready_count].resume = resume_of(frame);
    ready_count++;
}

static future *future_new(void) {
    future *f = (future *)calloc(1, sizeof(future));
    return f;
}

static void future_release(future *f) {
    free(f->waiters);
    free(f);
}

static void settle(future *f, int32_t state, int64_t bits) {
    if (f->state != PENDING) return;
    f->state = state;
    f->bits = bits;
    for (int64_t i = 0; i < f->waiter_count; i++) schedule(f->waiters[i]);
    f->waiter_count = 0;
}

int64_t *ppy_aio_frame_new(int64_t slots) {
    if (slots < 3) slots = 3;
    return (int64_t *)calloc((size_t)slots, sizeof(int64_t));
}

int64_t ppy_aio_spawn(int64_t *frame, ppy_aio_resume_fn resume) {
    future *f = future_new();
    frame[SLOT_SELF] = (int64_t)(intptr_t)f;
    frame[SLOT_STATE] = 0;
    if (frame_count == frame_room) {
        frame_room = frame_room ? frame_room * 2 : 16;
        frames = (registered *)realloc(frames, (size_t)frame_room * sizeof(registered));
    }
    frames[frame_count].frame = frame;
    frames[frame_count].resume = resume;
    frame_count++;
    f->frame = frame;
    return (int64_t)(intptr_t)f;
}

/* A coroutine runs once something waits for it -- an await, a start, a run --
   the way a Python coroutine object runs once awaited. */
static void start(future *f) {
    if (f->frame != NULL && !f->started) {
        f->started = 1;
        schedule(f->frame);
    }
}

void ppy_aio_start(int64_t handle) { start((future *)(intptr_t)handle); }

void ppy_aio_await(int64_t *frame, int64_t handle) {
    future *f = (future *)(intptr_t)handle;
    frame[SLOT_AWAITING] = handle;
    start(f);
    if (f->state != PENDING) {
        schedule(frame);
        return;
    }
    if (f->waiter_count == f->waiter_room) {
        f->waiter_room = f->waiter_room ? f->waiter_room * 2 : 2;
        f->waiters = (int64_t **)realloc(f->waiters, (size_t)f->waiter_room * sizeof(int64_t *));
    }
    f->waiters[f->waiter_count++] = frame;
}

int64_t ppy_aio_result(int64_t *frame) {
    future *f = (future *)(intptr_t)frame[SLOT_AWAITING];
    int64_t bits = f->bits;
    frame[SLOT_AWAITING] = 0;
    /* The awaited future is consumed: an awaiter is its only reader. */
    if (!f->consumed) {
        f->consumed = 1;
        future_release(f);
    }
    return bits;
}

static void finish(int64_t *frame, int32_t state, int64_t bits) {
    future *f = (future *)(intptr_t)frame[SLOT_SELF];
    settle(f, state, bits);
    forget_frame(frame);
    free(frame);
}

void ppy_aio_complete(int64_t *frame, int64_t bits) { finish(frame, DONE, bits); }

void ppy_aio_fail(int64_t *frame, int64_t code) { finish(frame, FAILED, code); }

/* -- timers ------------------------------------------------------------------------- */

static double now_seconds(void) {
#if PPY_AIO_EPOLL
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#else
    return 0.0;
#endif
}

int64_t ppy_aio_sleep(double seconds) {
    future *f = future_new();
    if (timer_count == timer_room) {
        timer_room = timer_room ? timer_room * 2 : 16;
        timers = (timer *)realloc(timers, (size_t)timer_room * sizeof(timer));
    }
    timers[timer_count].deadline = now_seconds() + (seconds > 0 ? seconds : 0);
    timers[timer_count].result = f;
    timer_count++;
    return (int64_t)(intptr_t)f;
}

static void expire_timers(double now) {
    for (int64_t i = 0; i < timer_count;) {
        if (timers[i].deadline <= now) {
            future *f = timers[i].result;
            timers[i] = timers[--timer_count];
            settle(f, DONE, 0);
        } else {
            i++;
        }
    }
}

static double next_deadline(void) {
    double best = -1;
    for (int64_t i = 0; i < timer_count; i++) {
        if (best < 0 || timers[i].deadline < best) best = timers[i].deadline;
    }
    return best;
}

/* -- sockets ------------------------------------------------------------------------ */

#if PPY_AIO_EPOLL

static int ensure_poller(void) {
    if (poller < 0) poller = epoll_create1(EPOLL_CLOEXEC);
    return poller;
}

static pending_io *pending_for(int fd) {
    if (fd < 0) return NULL;
    if (fd >= pending_room) {
        int64_t room = pending_room ? pending_room : 64;
        while (room <= fd) room *= 2;
        pending = (pending_io **)realloc(pending, (size_t)room * sizeof(pending_io *));
        memset(pending + pending_room, 0, (size_t)(room - pending_room) * sizeof(pending_io *));
        pending_room = room;
    }
    return pending[fd];
}

static int64_t watch(int fd, int kind, future *f, uint8_t *buffer, int64_t count, int64_t done) {
    if (ensure_poller() < 0) return -errno;
    pending_for(fd);
    pending_io *io = (pending_io *)calloc(1, sizeof(pending_io));
    io->kind = kind;
    io->result = f;
    io->buffer = buffer;
    io->count = count;
    io->done = done;
    pending[fd] = io;
    pending_count++;
    struct epoll_event event;
    memset(&event, 0, sizeof(event));
    event.events = (kind == OP_CONNECT || kind == OP_WRITE) ? EPOLLOUT : EPOLLIN;
    event.events |= EPOLLONESHOT;
    event.data.fd = fd;
    if (epoll_ctl(poller, EPOLL_CTL_ADD, fd, &event) < 0 && errno == EEXIST) {
        epoll_ctl(poller, EPOLL_CTL_MOD, fd, &event);
    }
    return 0;
}

static void unwatch(int fd) {
    pending_io *io = pending[fd];
    pending[fd] = NULL;
    pending_count--;
    free(io);
    epoll_ctl(poller, EPOLL_CTL_DEL, fd, NULL);
}

static char *host_string(const uint8_t *host, int64_t length) {
    char *text = (char *)malloc((size_t)length + 1);
    memcpy(text, host, (size_t)length);
    text[length] = 0;
    return text;
}

static int64_t open_socket(const uint8_t *host, int64_t length, int64_t port, int listening,
                           int64_t backlog) {
    char *node = host_string(host, length);
    char service[16];
    snprintf(service, sizeof(service), "%lld", (long long)port);
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (listening) hints.ai_flags = AI_PASSIVE;
    struct addrinfo *found = NULL;
    int status = getaddrinfo(length ? node : NULL, service, &hints, &found);
    free(node);
    if (status != 0 || found == NULL) return -EHOSTUNREACH;
    int fd = socket(found->ai_family, found->ai_socktype | SOCK_NONBLOCK | SOCK_CLOEXEC,
                    found->ai_protocol);
    if (fd < 0) {
        int saved = errno;
        freeaddrinfo(found);
        return -saved;
    }
    int64_t result = fd;
    if (listening) {
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (bind(fd, found->ai_addr, found->ai_addrlen) < 0 || listen(fd, (int)backlog) < 0) {
            result = -errno;
            close(fd);
        }
    } else if (connect(fd, found->ai_addr, found->ai_addrlen) < 0 && errno != EINPROGRESS) {
        result = -errno;
        close(fd);
    }
    freeaddrinfo(found);
    return result;
}

int64_t ppy_aio_listen(const uint8_t *host, int64_t length, int64_t port, int64_t backlog) {
    return open_socket(host, length, port, 1, backlog > 0 ? backlog : 16);
}

int64_t ppy_aio_port(int64_t socket_fd) {
    struct sockaddr_storage address;
    socklen_t size = sizeof(address);
    if (getsockname((int)socket_fd, (struct sockaddr *)&address, &size) < 0) return -errno;
    if (address.ss_family == AF_INET) return ntohs(((struct sockaddr_in *)&address)->sin_port);
    if (address.ss_family == AF_INET6) return ntohs(((struct sockaddr_in6 *)&address)->sin6_port);
    return -EAFNOSUPPORT;
}

void ppy_aio_close(int64_t socket_fd) {
    int fd = (int)socket_fd;
    if (fd < 0) return;
    if (fd < pending_room && pending[fd] != NULL) {
        settle(pending[fd]->result, DONE, -ECANCELED);
        unwatch(fd);
    }
    close(fd);
}

int64_t ppy_aio_accept(int64_t socket_fd) {
    future *f = future_new();
    int fd = (int)socket_fd;
    int accepted = accept4(fd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
    if (accepted >= 0) {
        settle(f, DONE, accepted);
    } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
        int64_t status = watch(fd, OP_ACCEPT, f, NULL, 0, 0);
        if (status < 0) settle(f, DONE, status);
    } else {
        settle(f, DONE, -errno);
    }
    return (int64_t)(intptr_t)f;
}

int64_t ppy_aio_connect(const uint8_t *host, int64_t length, int64_t port) {
    future *f = future_new();
    int64_t fd = open_socket(host, length, port, 0, 0);
    if (fd < 0) {
        settle(f, DONE, fd);
    } else {
        int64_t status = watch((int)fd, OP_CONNECT, f, NULL, 0, 0);
        if (status < 0) settle(f, DONE, status);
    }
    return (int64_t)(intptr_t)f;
}

int64_t ppy_aio_read(int64_t socket_fd, uint8_t *buffer, int64_t count) {
    future *f = future_new();
    int fd = (int)socket_fd;
    ssize_t got = read(fd, buffer, (size_t)count);
    if (got >= 0) {
        settle(f, DONE, got);
    } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
        int64_t status = watch(fd, OP_READ, f, buffer, count, 0);
        if (status < 0) settle(f, DONE, status);
    } else {
        settle(f, DONE, -errno);
    }
    return (int64_t)(intptr_t)f;
}

int64_t ppy_aio_write(int64_t socket_fd, const uint8_t *buffer, int64_t count) {
    future *f = future_new();
    int fd = (int)socket_fd;
    int64_t done = 0;
    while (done < count) {
        ssize_t put = write(fd, buffer + done, (size_t)(count - done));
        if (put >= 0) {
            done += put;
        } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
            int64_t status = watch(fd, OP_WRITE, f, (uint8_t *)buffer, count, done);
            if (status < 0) settle(f, DONE, status);
            return (int64_t)(intptr_t)f;
        } else {
            settle(f, DONE, -errno);
            return (int64_t)(intptr_t)f;
        }
    }
    settle(f, DONE, done);
    return (int64_t)(intptr_t)f;
}

static void serve_event(int fd) {
    pending_io *io = pending[fd];
    if (io == NULL) return;
    future *f = io->result;
    if (io->kind == OP_ACCEPT) {
        int accepted = accept4(fd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (accepted < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            struct epoll_event event;
            memset(&event, 0, sizeof(event));
            event.events = EPOLLIN | EPOLLONESHOT;
            event.data.fd = fd;
            epoll_ctl(poller, EPOLL_CTL_MOD, fd, &event);
            return;
        }
        unwatch(fd);
        settle(f, DONE, accepted >= 0 ? accepted : -errno);
    } else if (io->kind == OP_CONNECT) {
        int error = 0;
        socklen_t size = sizeof(error);
        getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &size);
        unwatch(fd);
        if (error) {
            close(fd);
            settle(f, DONE, -error);
        } else {
            settle(f, DONE, fd);
        }
    } else if (io->kind == OP_READ) {
        ssize_t got = read(fd, io->buffer, (size_t)io->count);
        if (got < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            struct epoll_event event;
            memset(&event, 0, sizeof(event));
            event.events = EPOLLIN | EPOLLONESHOT;
            event.data.fd = fd;
            epoll_ctl(poller, EPOLL_CTL_MOD, fd, &event);
            return;
        }
        unwatch(fd);
        settle(f, DONE, got >= 0 ? got : -errno);
    } else if (io->kind == OP_WRITE) {
        while (io->done < io->count) {
            ssize_t put = write(fd, io->buffer + io->done, (size_t)(io->count - io->done));
            if (put >= 0) {
                io->done += put;
            } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
                struct epoll_event event;
                memset(&event, 0, sizeof(event));
                event.events = EPOLLOUT | EPOLLONESHOT;
                event.data.fd = fd;
                epoll_ctl(poller, EPOLL_CTL_MOD, fd, &event);
                return;
            } else {
                int saved = errno;
                unwatch(fd);
                settle(f, DONE, -saved);
                return;
            }
        }
        int64_t done = io->done;
        unwatch(fd);
        settle(f, DONE, done);
    }
}

#else /* not Linux: the runtime refuses, and the language's Python path runs */

int64_t ppy_aio_listen(const uint8_t *host, int64_t length, int64_t port, int64_t backlog) {
    (void)host; (void)length; (void)port; (void)backlog;
    return -ENOSYS;
}
void ppy_aio_close(int64_t socket_fd) { (void)socket_fd; }
int64_t ppy_aio_port(int64_t socket_fd) { (void)socket_fd; return -ENOSYS; }
static int64_t refused(void) {
    future *f = future_new();
    settle(f, DONE, -ENOSYS);
    return (int64_t)(intptr_t)f;
}
int64_t ppy_aio_accept(int64_t s) { (void)s; return refused(); }
int64_t ppy_aio_connect(const uint8_t *h, int64_t l, int64_t p) { (void)h; (void)l; (void)p; return refused(); }
int64_t ppy_aio_read(int64_t s, uint8_t *b, int64_t c) { (void)s; (void)b; (void)c; return refused(); }
int64_t ppy_aio_write(int64_t s, const uint8_t *b, int64_t c) { (void)s; (void)b; (void)c; return refused(); }
static void serve_event(int fd) { (void)fd; }

#endif

/* -- the loop ----------------------------------------------------------------------- */

static void run_ready(void) {
    while (ready_count > 0) {
        frame_entry entry = ready[0];
        memmove(ready, ready + 1, (size_t)(ready_count - 1) * sizeof(frame_entry));
        ready_count--;
        if (entry.resume != NULL) entry.resume(entry.frame);
    }
}

int32_t ppy_aio_step(int64_t timeout_ms) {
    run_ready();
    if (ready_count == 0 && timer_count == 0 && pending_count == 0) return 0;
    double now = now_seconds();
    double deadline = next_deadline();
    int64_t wait = timeout_ms;
    if (deadline >= 0) {
        int64_t until = (int64_t)((deadline - now) * 1000.0);
        if (until < 0) until = 0;
        if (wait < 0 || until < wait) wait = until;
    }
    if (ready_count > 0) wait = 0;
#if PPY_AIO_EPOLL
    if (pending_count > 0 && ensure_poller() >= 0) {
        struct epoll_event events[64];
        int got = epoll_wait(poller, events, 64, (int)(wait < 0 ? -1 : wait));
        for (int i = 0; i < got; i++) serve_event(events[i].data.fd);
    } else if (wait > 0) {
        struct timespec ts;
        ts.tv_sec = wait / 1000;
        ts.tv_nsec = (wait % 1000) * 1000000L;
        nanosleep(&ts, NULL);
    }
#endif
    expire_timers(now_seconds());
    run_ready();
    return (ready_count > 0 || timer_count > 0 || pending_count > 0) ? 1 : 0;
}

int32_t ppy_aio_run(int64_t handle) {
    future *f = (future *)(intptr_t)handle;
    start(f);
    while (f->state == PENDING) {
        if (ppy_aio_step(-1) == 0 && f->state == PENDING) return 2;
    }
    return f->state == DONE ? 0 : 1;
}

int32_t ppy_aio_state(int64_t handle) { return ((future *)(intptr_t)handle)->state; }

int64_t ppy_aio_failure(int64_t handle) { return ((future *)(intptr_t)handle)->bits; }

int64_t ppy_aio_take(int64_t handle) {
    future *f = (future *)(intptr_t)handle;
    int64_t bits = f->bits;
    if (!f->consumed) {
        f->consumed = 1;
        future_release(f);
    }
    return bits;
}

int32_t ppy_aio_step_started(int64_t handle) {
    start((future *)(intptr_t)handle);
    return ppy_aio_step(0);
}

const char *ppy_aio_platform(void) {
#if PPY_AIO_EPOLL
    return "epoll";
#else
    return "none";
#endif
}
