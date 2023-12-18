import datetime
import re

from scipy.ndimage import binary_fill_holes
from skimage.draw import polygon_perimeter

import numpy as np
from aocd.models import Puzzle

YEAR = 2023
DAY = 18

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    corners = [0 + 0]
    dirs = {
        "R": 0 + 1j,
        "U": -1 + 0j,
        "L": 0 - 1j,
        "D": 1 + 0j,
    }
    curr = corners[0]
    for line in data.splitlines():
        dir, steps, _ = line.split(" ")
        curr += dirs[dir] * int(steps)
        corners.append(curr)
    corners.append(corners[0])

    edge = 0
    for i in range(1, len(corners)):
        edge +=abs(corners[i - 1] - corners[i])

    area = 0
    for i in range(1, len(corners)):
        area += int(corners[i].real) * (int(corners[i].imag) - int(corners[i-1].imag))
    area = abs(area)

    return area + int(edge) // 2 + 1


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 48400


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
