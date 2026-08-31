"""Single-source shortest paths: https://www.acmicpc.net/problem/1753

Input:  V E on the first line, the source K on the second, then E lines of
        `u v w` (a directed edge from u to v of weight w).
Output: the sum of the reachable distances, so one line stands in for V.
"""

import array

INFINITY = 1 << 60


def sift_down(keys, nodes, size, start):
    position = start
    while True:
        left = 2 * position + 1
        if left >= size:
            break
        best = left
        right = left + 1
        if right < size and keys[right] < keys[left]:
            best = right
        if keys[position] <= keys[best]:
            break
        key = keys[position]
        node = nodes[position]
        keys[position] = keys[best]
        nodes[position] = nodes[best]
        keys[best] = key
        nodes[best] = node
        position = best
    return position


def sift_up(keys, nodes, start):
    position = start
    while position > 0:
        parent = (position - 1) // 2
        if keys[parent] <= keys[position]:
            break
        key = keys[position]
        node = nodes[position]
        keys[position] = keys[parent]
        nodes[position] = nodes[parent]
        keys[parent] = key
        nodes[parent] = node
        position = parent
    return position


def dijkstra(
    offsets,
    targets,
    weights,
    distance,
    keys,
    nodes,
    count,
    source,
):
    for i in range(count):
        distance[i] = INFINITY
    distance[source] = 0
    keys[0] = 0
    nodes[0] = source
    size = 1
    while size > 0:
        best = keys[0]
        node = nodes[0]
        size -= 1
        keys[0] = keys[size]
        nodes[0] = nodes[size]
        sift_down(keys, nodes, size, 0)
        if best <= distance[node]:
            for edge in range(offsets[node], offsets[node + 1]):
                neighbour = targets[edge]
                relaxed = best + weights[edge]
                if relaxed < distance[neighbour]:
                    distance[neighbour] = relaxed
                    keys[size] = relaxed
                    nodes[size] = neighbour
                    size += 1
                    sift_up(keys, nodes, size - 1)
    total = 0
    for i in range(count):
        if distance[i] < INFINITY:
            total += distance[i]
    return total


def build_csr(
    fields,
    offsets,
    targets,
    weights,
    cursor,
    count,
    edges,
):
    """Turn the `u v w` triples into compressed adjacency, in place."""
    for e in range(edges):
        offsets[fields[3 + e * 3] + 1] += 1
    for i in range(count):
        offsets[i + 1] += offsets[i]
    for i in range(count):
        cursor[i] = offsets[i]
    for e in range(edges):
        head = fields[3 + e * 3]
        at = cursor[head]
        targets[at] = fields[4 + e * 3]
        weights[at] = fields[5 + e * 3]
        cursor[head] = at + 1
    return offsets[count]


def generated():
    """The stand-in graph, in the same `V E K u v w ...` order as the input."""
    count = 200000
    degree = 6
    fields = [count, count * degree, 0]
    for i in range(count):
        for k in range(degree):
            fields.append(i)
            fields.append((i * 7919 + k * 104729 + 1) % count)
            fields.append((i * 31 + k * 17) % 1000 + 1)
    return fields


def read_fields():
    """The judge's input, or a generated stand-in when nothing is piped in."""
    try:
        count, edges = map(int, input().split())
    except EOFError:
        return array.array("q", generated())
    fields = array.array("q", [count, edges, int(input())])
    triples = array.array("q", [0] * (edges * 3))
    for i in range(edges * 3):
        triples[i] = int(input())
    fields.extend(triples)
    return fields


def main():
    fields = read_fields()
    count = fields[0]
    edges = fields[1]
    source = fields[2]
    offsets = array.array("q", [0] * (count + 1))
    cursor = array.array("q", [0] * count)
    targets = array.array("q", [0] * edges)
    weights = array.array("q", [0] * edges)
    distance = array.array("q", [0] * count)
    keys = array.array("q", [0] * (edges + 1))
    nodes = array.array("q", [0] * (edges + 1))

    build_csr(fields, offsets, targets, weights, cursor, count, edges)
    print(dijkstra(offsets, targets, weights, distance, keys, nodes, count, source))


main()
