import ast
import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, lim0, lim1):
    positions = []
    velocities = []
    for line in data.splitlines():
        pos, v = line.split("@")
        pos = ast.literal_eval(pos)
        v = ast.literal_eval(v.strip())
        positions.extend(list(pos))
        velocities.extend(list(v))
    positions = np.array(positions, dtype=int)
    positions.shape = (-1, 3)
    velocities = np.array(velocities, dtype=int)
    velocities.shape = (-1, 3)

    positions = positions[:, :2]
    velocities = velocities[:, :2]

    positions -= lim0
    lim1 -= lim0
    lim0 -= lim0

    positions = positions.astype(float)
    positions /= lim1
    lim1 /= lim1

    s = 0
    for p, v in zip(positions, velocities):
        p1 = p
        p2 = p + v
        p3 = positions
        p4 = positions + velocities
        x = ((p1[0]*p2[1]-p1[1]*p2[0])*(p3[:,0]-p4[:,0]) - (p1[0] - p2[0]) *(p3[:,0]*p4[:,1]-p3[:,1]*p4[:,0]))/((p1[0] - p2[0]) *(p3[:,1] - p4[:,1]) -(p1[1] - p2[1]) *(p3[:,0] - p4[:,0]))
        y = ((p1[0]*p2[1]-p1[1]*p2[0])*(p3[:,1]-p4[:,1]) - (p1[1] - p2[1]) *(p3[:,0]*p4[:,1]-p3[:,1]*p4[:,0]))/((p1[0] - p2[0]) *(p3[:,1] - p4[:,1]) -(p1[1] - p2[1]) *(p3[:,0] - p4[:,0]))
        tmp = (x >= lim0) & (x <= lim1) & (y >= lim0) & (y <= lim1)
        t1 = (x - p[0]) / v[0]
        t2 = (y - p[1]) / v[1]
        t3 = (x - p3[:,0]) / velocities[:, 0]
        t4 = (y - p3[:,1]) / velocities[:, 1]
        #breakpoint()
        s += sum(tmp & (t1 > 0) & (t2 > 0) & (t3 > 0) & (t4 > 0))
    return s // 2


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data,7,27)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert example_answer == 2
answer = a(puzzle.input_data, 200000000000000, 400000000000000)
print("a:", answer)
assert answer > 11793
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
