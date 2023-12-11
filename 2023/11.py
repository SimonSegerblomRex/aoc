import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    grid = np.vstack(
        [np.frombuffer(row.encode(), dtype=np.uint8) for row in data.splitlines()]
    )
    grid[grid == ord(".")] = 0
    grid[grid == ord("#")] = 1
    #grid = np.pad(grid, 1, constant_values=ord("."))
    zero_rows = []
    for i, row in enumerate(range(grid.shape[0])):
        row = grid[i, :]
        if row.any():
            continue
        zero_rows.append(i)
    zero_cols = []
    for j, col in enumerate(range(grid.shape[1])):
        col = grid[:, j]
        if col.any():
            continue
        zero_cols.append(j)
    grid = np.insert(grid, zero_rows, 0, axis=0)
    grid = np.insert(grid, zero_cols, 0, axis=1)
    galaxies = np.nonzero(grid)
    from collections import defaultdict
    distances = defaultdict(lambda:np.inf)
    for i, galaxy_from in enumerate(zip(*galaxies)):
        for j, galaxy_to in enumerate(zip(*galaxies)):
            d = np.abs(galaxy_from[0] - galaxy_to[0]) + np.abs(galaxy_from[1] - galaxy_to[1])
            if d:
                idx = tuple(sorted((i, j)))
                distances[idx] = min(distances[idx], d)
    return sum(distances.values())
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
    breakpoint()


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
