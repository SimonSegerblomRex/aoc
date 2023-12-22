import ast

import numpy as np
from aocd.models import Puzzle

YEAR = 2023
DAY = 22

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def get_neighbours(data):
    bricks = []
    max_z = 0
    for line in data.splitlines():
        c0, c1 = line.split("~")
        c0 = list(ast.literal_eval(c0))
        c1 = list(ast.literal_eval(c1))
        bricks.append([c0, c1])
        max_z = max(max_z, c0[2], c1[2])

    z = np.zeros((10, 10, max_z), dtype=int)
    z[:, :, 0] = 1  # ground
    bricks = sorted(bricks, key=lambda b: min(b[0][2], b[1][2]))

    n = 1
    for c0, c1 in bricks:
        n += 1
        lowest_z = z.argmax(axis=2)[c0[0] : c1[0] + 1, c0[1] : c1[1] + 1].max() + 1
        while True:
            if (
                z[
                    c0[0] : c1[0] + 1,
                    c0[1] : c1[1] + 1,
                    lowest_z : lowest_z + c1[2] - c0[2] + 1,
                ].sum()
                > 0
            ):
                lowest_z += 1
            else:
                break
        z[
            c0[0] : c1[0] + 1,
            c0[1] : c1[1] + 1,
            lowest_z : lowest_z + c1[2] - c0[2] + 1,
        ] = n
        c0[2] = lowest_z
        c1[2] = lowest_z + c1[2] - c0[2]

    neighbours = {}
    for n in range(2, n + 1):
        neigh = set()
        for c in zip(*np.nonzero(z == n)):
            c = (c[0], c[1], c[2] + 1)
            if z[c] > n:
                neigh.add(z[c])
        neighbours[n] = neigh
    return neighbours


def a(data):
    neighbours = get_neighbours(data)
    s = 0
    for n, neigh in neighbours.items():
        others = neighbours.copy()
        del others[n]
        tmp = neigh.difference(*(list(s) for s in others.values()))
        if not tmp:
            s += 1
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 482


# Part b
def disintegrate(blocks, neighbours):
    s = set()
    neigh = set()
    for n in blocks:
        neigh |= neighbours.pop(n)
    if not neigh:
        return s
    neigh ^= neigh.intersection(set.union(*(s for s in neighbours.values())))
    if neigh:
        s |= neigh
        s |= disintegrate(neigh, neighbours.copy())
    return s


def b(data):
    neighbours = get_neighbours(data)
    s = 0
    for n in neighbours:
        s += len(disintegrate(set([n]), neighbours.copy()))
    return s


example_answer = b(example.input_data)
print(f"Example answer: {example_answer} (expecting: {7})")
assert example_answer == 7
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 103010
