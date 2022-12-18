import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    data =  [np.fromstring(line, dtype=int, sep=",") for line in data.splitlines()]
    data = np.vstack(data)
    xx = data[:, 1]
    yy = data[:, 0]
    zz = data[:, 2]
    width = np.max(xx) + 1
    height = np.max(yy) + 1
    depth = np.max(zz) + 1
    grid = np.zeros((width, height, depth), dtype=int)
    for x, y, z in zip(xx, yy, zz):
        grid[x, y, z] = 1
    grid = np.pad(grid, 1)
    return np.sum(np.diff(grid, axis=0) > 0) + np.sum(np.diff(grid[::-1, :, :], axis=0) > 0) + np.sum(np.diff(grid, axis=1) > 0) + np.sum(np.diff(grid[:, ::-1, :], axis=1) > 0) + np.sum(np.diff(grid, axis=2) > 0) + np.sum(np.diff(grid[:, :, ::-1], axis=2) > 0)


example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 64
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 3326


# Part b
def b(data):
    exit()

example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 58
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
