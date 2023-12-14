import re

import numpy as np
from aocd.models import Puzzle

YEAR = 2023
DAY = 14

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def text_to_numpy(text):
    grid = np.vstack(
        [np.frombuffer(row.encode(), dtype="|S1") for row in text.splitlines()]
    )
    return grid


def tilt_row(row):
    matches = re.finditer(r"([O.]+)", row)
    out = np.array(list(row.replace("O", ".")))
    for m in matches:
        n = m.group(1).count("O")
        if n:
            out[m.start() : m.start() + n] = "O"
    return "".join(out)


def rotate_grid(grid, k):
    grid = text_to_numpy(grid)
    grid = np.rot90(grid, k)
    return "\n".join(row.tobytes().decode() for row in grid)


def tilt_grid(grid):
    grid = rotate_grid(grid, 1)
    new_rows = []
    for row in grid.splitlines():
        new_row = tilt_row(row)
        new_rows.append(new_row)
    new_grid = "\n".join(new_rows)
    return rotate_grid(new_grid, -1)


def score_grid(grid):
    rows = grid.splitlines()
    scores = range(len(rows), 0, -1)
    return sum(r.count("O") * s for r, s in zip(rows, scores))


def a(data):
    grid = tilt_grid(data)
    return score_grid(grid)


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 108759


# Part b
def b(data, rotations, cache=True):
    grid_cache = {}
    curr_grid = data
    for i in range(rotations):
        curr_grid = tilt_grid(curr_grid)
        if cache and not (i + 1) % 4:
            if curr_grid in grid_cache:
                print(i - grid_cache[curr_grid])
                to_go = (rotations - i) % (i - grid_cache[curr_grid])
                tmp = b(curr_grid, to_go, cache=False)
                return tmp
            grid_cache[curr_grid] = i
            if 0:
                # Debug prints
                print(i)
                tmp_grid = rotate_grid(curr_grid, i % 4)
                print(tmp_grid)
                breakpoint()
        curr_grid = rotate_grid(curr_grid, -1)
    return score_grid(curr_grid)


example_answer = b(example.input_data, 1000000000 * 4)
print(f"Example answer: {example_answer} (expecting: {64})")
assert example_answer == 64
answer = b(puzzle.input_data, 1000000000 * 4)
print("b:", answer)
assert answer == 89089
