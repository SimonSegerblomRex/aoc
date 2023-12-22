import ast
import datetime
import re
from scipy.signal import convolve

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    bricks = []
    max_z = 0
    for line in data.splitlines():
        c0, c1 = line.split("~")
        c0 = list(ast.literal_eval(c0))
        c1 = list(ast.literal_eval(c1))
        bricks.append([c0, c1])
        max_z = max(max_z, c0[2])
        max_z = max(max_z, c1[2])

    z = np.zeros((12+10,12+10,max_z+10), dtype=int)
    z[:,:,0] = 1 #ground
    bricks = sorted(bricks, key=lambda b: min(b[0][2], b[1][2]))

    n = 1
    for c0, c1 in bricks:
        n += 1
        lowest_z = z.argmax(axis=2)[c0[0]:c1[0]+1,c0[1]:c1[1]+1].max() + 1
        if 1:
            print(lowest_z)
            while True:
                if z[c0[0]:c1[0]+1,c0[1]:c1[1]+1,lowest_z:lowest_z+c1[2]-c0[2]+1].sum() > 0:
                    lowest_z += 1
                else:
                    break
            print(lowest_z)
            print("")
        z[c0[0]:c1[0]+1,c0[1]:c1[1]+1,lowest_z:lowest_z+c1[2]-c0[2]+1] = n
        c0[2] = lowest_z
        c1[2] = lowest_z + c1[2] - c0[2]

    neighbours = {}
    for n in range(2, n + 1):
        neigh = set()
        for c in zip(*np.nonzero(z==n)):
            c = (c[0], c[1], c[2] + 1)
            if z[c] > n:
                neigh.add(z[c])
        neighbours[n] = neigh

    s = 0
    for n, neigh in neighbours.items():
        others = neighbours.copy()
        del others[n]
        tmp = neigh.difference(*(list(s) for s in others.values()))
        if not tmp:
            print(n)
            s += 1

    breakpoint()
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer < 503
puzzle.answer_a = answer


# Part b
def b(data):
    breakpoint()


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
