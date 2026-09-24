"""The same problems with Python's own containers: list, deque, heapq, dict, bisect."""

import bisect
import heapq
from collections import deque

NODES = 200000
EDGES = 4


def target(node, k):
    return (node * 7919 + k * 104729 + 13) % NODES


def weight(node, k):
    return (node * 31 + k * 17) % 97 + 1


def shortest_paths(source):
    dist = [-1] * NODES
    frontier = [source]
    dist[source] = 0
    while frontier:
        packed = heapq.heappop(frontier)
        node, here = packed % NODES, packed // NODES
        if here > dist[node]:
            continue
        for k in range(EDGES):
            other = target(node, k)
            through = here + weight(node, k)
            if dist[other] < 0 or through < dist[other]:
                dist[other] = through
                heapq.heappush(frontier, through * NODES + other)
    return sum(dist)


def hops(source):
    seen = [-1] * NODES
    queue = deque([source])
    seen[source] = 0
    farthest = 0
    while queue:
        node = queue.popleft()
        for k in range(EDGES):
            other = target(node, k)
            if seen[other] < 0:
                seen[other] = seen[node] + 1
                farthest = max(farthest, seen[other])
                queue.append(other)
    return farthest


def josephus(people, step):
    circle = list(range(people))
    at = 0
    while len(circle) > 1:
        at = (at + step - 1) % len(circle)
        circle.pop(at)
    return circle[0]


def repeats(count):
    seen = {}
    for i in range(count):
        key = (i * i + 7 * i) % 65537
        seen[key] = seen.get(key, 0) + 1
    return len(seen) * 100 + max(seen.values())


def closest_gaps(count):
    marks = []
    best = 1 << 40
    for i in range(count):
        value = (i * 2654435761) % 1000003
        at = bisect.bisect_left(marks, value)
        if at > 0:
            best = min(best, value - marks[at - 1])
        if at < len(marks):
            best = min(best, marks[at] - value)
        marks.insert(at, value)
    return best


print(shortest_paths(0), hops(0))
print(josephus(100000, 7), repeats(2000000), closest_gaps(200000))
