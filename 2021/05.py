import re

import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 5

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    data = re.findall("(\d+),(\d+) -> (\d+),(\d+)", data)
    data = [[int(x) for x in p] for p in data]
    grid = np.zeros((1000, 1000))
    for x0, y0, x1, y1 in data:
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if (x0 == x1) or (y0 == y1):
            grid[y0 : y1 + 1, x0 : x1 + 1] += 1
    print(grid[:10, :10])
    return (grid > 1).sum()


assert a(puzzle.example_data) == 5
answer = a(puzzle.input_data)
print("a:", answer)


# Part b
def b(data):
    data = re.findall("(\d+),(\d+) -> (\d+),(\d+)", data)
    data = [[int(x) for x in p] for p in data]
    grid = np.zeros((1000, 1000))
    for x0, y0, x1, y1 in data:
        if (x0 == x1) or (y0 == y1):
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            grid[y0 : y1 + 1, x0 : x1 + 1] += 1
        else:
            # Assume 45 degree slope
            x0s, x1s = sorted((x0, x1))
            y0s, y1s = sorted((y0, y1))
            xx = list(range(x0s, x1s + 1))
            yy = list(range(y0s, y1s + 1))
            if x0 > x1:
                xx = xx[::-1]
            if y0 > y1:
                yy = yy[::-1]
            for x, y in zip(xx, yy):
                grid[y, x] += 1
    print(grid[:10, :10])
    return (grid > 1).sum()


assert b(puzzle.example_data) == 12
answer = b(puzzle.input_data)
print("b:", answer)
