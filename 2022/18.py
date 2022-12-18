import numpy as np
from aocd.models import Puzzle
from scipy.ndimage import binary_fill_holes


YEAR = 2022
DAY = 18

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def parse_data(data):
    data = np.vstack(
        [np.fromstring(line, dtype=int, sep=",") for line in data.splitlines()]
    )
    xx, yy, zz = np.hsplit(data, 3)
    width = np.max(xx) + 1
    height = np.max(yy) + 1
    depth = np.max(zz) + 1
    grid = np.zeros((width, height, depth), dtype=int)
    grid[xx, yy, zz] = 1
    grid = np.pad(grid, 1)
    return grid


def calc_surface_area(grid):
    return sum((np.diff(grid, axis=i) != 0).sum() for i in range(grid.ndim))


def a(data):
    grid = parse_data(data)
    return calc_surface_area(grid)


example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 64
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 3326


# Part b
def b(data):
    grid = parse_data(data)
    filled = binary_fill_holes(grid)
    trapped = filled - grid
    return calc_surface_area(grid) - calc_surface_area(trapped)


example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 58
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 1996
