import re

import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 13

puzzle = Puzzle(year=YEAR, day=DAY)

EXAMPLE_DATA = """6,10
0,14
9,10
0,3
10,4
4,11
6,0
6,12
4,1
0,13
10,12
3,4
3,0
8,4
1,10
2,14
8,10
9,0

fold along y=7
fold along x=5"""


# Part a
def a(data):
    coords, instructions = data.split("\n\n")
    coords = np.fromstring(coords.replace("\n", ","), sep=",", dtype=int).reshape(-1, 2)
    instructions = re.findall("(x|y)=(\d+)", instructions)

    max_x, max_y = coords.max(axis=0)
    grid = np.full((max_x + 1, max_y + 1), False)
    grid[coords[:, 0], coords[:, 1]] = 1
    for axis, coord in instructions:
        coord = int(coord)
        if axis == "y":
            l1 = grid[:, :coord]
            l2 = grid[:, coord + 1:][:, ::-1]
            l2 = np.pad(l2, ((0, 0), (l1.shape[1] - l2.shape[1], 0)))
            grid = l1 | l2
        else:
            l1 = grid[:coord, :]
            l2 = grid[coord + 1:, :][::-1, :]
            l2 = np.pad(l2, ((l1.shape[0] - l2.shape[0], 0), (0, 0)))
            grid = l1 | l2
        return grid.sum()


example_answer = a(EXAMPLE_DATA)
print(example_answer)
assert example_answer == 17
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 706


# Part b
def b(data):
    exit()


example_answer = b(EXAMPLE_DATA)
print(example_answer)
assert example_answer == ...
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
