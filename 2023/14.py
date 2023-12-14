import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    #grid = np.vstack(
    #    [np.frombuffer(row.encode(), dtype=np.uint8) for row in data.splitlines()]
    #)
    #grid[grid == ord(".")] = 0
    #grid[grid == ord("O")] = 1
    #grid[grid == ord("#")] = 1
    if 0:
        first_square_rock = np.argmax(grid==ord("#"), axis=0)
        for j in range(grid.shape[1]):
            grid[first_square_rock[j]:, j] = 0
        rocks = grid.sum(axis=0)
        height, width = grid.shape
        total = 0
        for j in range(grid.shape[1]):
            tmp = int(rocks[j])
            total += sum(range(height, height - tmp, -1))
        return total
    grid = np.vstack(
        [np.frombuffer(row.encode(), dtype='|S1') for row in data.splitlines()]
    )
    total = 0
    for row in grid.T:
        row = row.tostring().decode()
        hmm = re.finditer(r"([O.]+)", row)
        s = 0
        l = len(row)
        for m in hmm:
            s += sum(range(l - m.start(), l - m.start() - m.group(1).count("O"), -1))
        total += s
    return total


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
