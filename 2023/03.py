import datetime
import re

import numpy as np
from aocd.models import Puzzle
from scipy.signal import convolve2d
from scipy.ndimage import watershed_ift

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    try:
        s = 0
        ss = 0
        grid = np.vstack(
            [np.frombuffer(n.encode(), dtype=np.uint8) - ord("0") for n in data.split("\n")]
        )
        symbols = np.logical_and(grid != 254, grid > 9)
        kernel = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ])
        hack = convolve2d(
            symbols.astype(np.uint8),
            kernel,
            mode="same",
            boundary="fill",
        ) > 0
        digits = grid < 10
        bad_digits = np.logical_and(digits, hack)
        background = grid > 9
        tmp = np.ones(grid.shape, dtype=np.uint8) * 255
        tmp[digits] = 0
        tmp[bad_digits] = 1
        markers = bad_digits.astype(int)
        markers[background] = -1
        ww = watershed_ift(tmp, markers)
        bad_digits = ww > 0
        good_digits = np.logical_and(digits, ~bad_digits)
        for row, d in zip(grid, digits):
            row[~d] = 0
            tmp = "".join(str(c) for c in row)
            ss += sum([int(e) if e else 0 for e in tmp.split("0")])
        for row, good in zip(grid, good_digits):
            row[~good] = 0
            tmp = "".join(str(c) for c in row)
            s += sum([int(e) if e else 0 for e in tmp.split("0")])
        #for row, mask in zip(grid, hack):
        #    breakpoint()
        return ss - s
    except:
        breakpoint()


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    print(data)
    breakpoint()


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
