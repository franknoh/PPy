/* The PPY native async runtime: frames, futures, timers, and sockets (spec 77). */
#ifndef PPY_AIO_H
#define PPY_AIO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A coroutine's resume function: runs the frame to its next suspension. */
typedef void (*ppy_aio_resume_fn)(int64_t *frame);

/* Frames: `slots` words, zeroed; the first three are the runtime's
   (the frame's own future, the future it awaits, its state). */
int64_t *ppy_aio_frame_new(int64_t slots);
int64_t ppy_aio_spawn(int64_t *frame, ppy_aio_resume_fn resume);
void ppy_aio_start(int64_t future);
void ppy_aio_await(int64_t *frame, int64_t future);
int64_t ppy_aio_result(int64_t *frame);
void ppy_aio_complete(int64_t *frame, int64_t bits);
void ppy_aio_fail(int64_t *frame, int64_t code);

/* Operations that complete later; each returns a future. */
int64_t ppy_aio_sleep(double seconds);
int64_t ppy_aio_accept(int64_t socket);
int64_t ppy_aio_connect(const uint8_t *host, int64_t length, int64_t port);
int64_t ppy_aio_read(int64_t socket, uint8_t *buffer, int64_t count);
int64_t ppy_aio_write(int64_t socket, const uint8_t *buffer, int64_t count);

/* Immediate operations. */
int64_t ppy_aio_listen(const uint8_t *host, int64_t length, int64_t port, int64_t backlog);
int64_t ppy_aio_port(int64_t socket);
void ppy_aio_close(int64_t socket);

/* The loop. `run` drives it until `future` completes: 0 done, 1 failed,
   2 nothing left to wait for. `step` runs one turn with a timeout in
   milliseconds (negative: until something happens); it returns 1 while
   work remains, 0 when the loop is idle. */
int32_t ppy_aio_run(int64_t future);
int32_t ppy_aio_step(int64_t timeout_ms);
int32_t ppy_aio_step_started(int64_t future);
int32_t ppy_aio_state(int64_t future);
int64_t ppy_aio_take(int64_t future);
int64_t ppy_aio_failure(int64_t future);
const char *ppy_aio_platform(void);

#ifdef __cplusplus
}
#endif

#endif
