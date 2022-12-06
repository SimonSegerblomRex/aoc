import re

import numpy as np
from scipy.ndimage import uniform_filter
from aocd.models import Puzzle

YEAR = 2021
DAY = 11

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    grid = np.fromstring(data.replace("\n", ""), dtype=np.uint8, sep="").reshape(
        10, 10
    ) - ord("0")
    flashes = 0
    for _ in range(100):
        grid += 1
        while True:
            flash = grid > 9
            if not flash.any():
                break
            grid[flash] = 0
            hack = (
                uniform_filter(flash.astype(float), size=3, mode="constant", cval=0) * 9
            ).astype(np.uint8)
            grid[grid > 0] += hack[grid > 0]
        print(grid)
        flashes += (grid == 0).sum()
    return flashes


example_answer = a(puzzle.example_data)
print(example_answer)
#assert example_answer == 1656  # Wrong in the example..?
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    exit()


example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == ...
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
