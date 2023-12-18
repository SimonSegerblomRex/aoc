import datetime
import re

from scipy.ndimage import binary_fill_holes
from skimage.draw import polygon_perimeter

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    corners = [0 + 0j]
    dirs = {
        "R": 0 + 1j,
        "U": -1 + 0j,
        "L": 0 - 1j,
        "D": 1 + 0j,
    }
    curr = 0 + 0j
    for line in data.splitlines():
        dir, steps, _ = line.split(" ")
        curr += dirs[dir] * int(steps)
        corners.append(curr)

    min_x = int(min(c.real for c in corners))
    min_y = int(min(c.imag for c in corners))
    corners = [c - complex(min_x, min_y) for c in corners]
    height = int(max(c.real for c in corners)) + 1
    width = int(max(c.imag for c in corners)) + 1
    print(height, width)
    rows = [int(c.real) for c in corners]
    cols = [int(c.imag) for c in corners]
    tmp = np.zeros((height, width), dtype=np.uint8)
    rr, cc = polygon_perimeter(rows, cols, shape=tmp.shape, clip=True)
    tmp[rr, cc] = 1
    tmp = binary_fill_holes(tmp)
    #tmp = tmp[::2, ::2]
    return tmp.sum()
    if 0:
        #color = [int(c, 16) for c in (color[2:4], color[5:7], color[6:8])]
        min_x = int(min(c.real for c in corners))
        min_y = int(min(c.imag for c in corners))
        corners = [c - complex(min_x, min_y) for c in corners]
        height = int(max(c.real for c in corners)) + 1
        width = int(max(c.imag for c in corners)) + 1
        tmp = np.zeros((height, width))
        for i in range(len(corners)-1):
            row_0 = int(corners[i].real)
            row_1 = int(corners[i+1].real)
            col_0 = int(corners[i].imag)
            col_1 = int(corners[i+1].imag)
            row_0, row_1 = sorted((row_0, row_1))
            col_0, col_1 = sorted((col_0, col_1))
            tmp[row_0:row_1+1, col_0:col_1+1] = 1
    breakpoint()


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer > 22793
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
