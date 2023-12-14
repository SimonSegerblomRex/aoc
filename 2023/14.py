import re
from functools import cache

import numpy as np
from aocd.models import Puzzle

YEAR = 2023
DAY = 14

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
@cache
def text_to_numpy(text):
    grid = np.vstack(
        [np.frombuffer(row.encode(), dtype="|S1") for row in text.splitlines()]
    )
    return grid

@cache
def compute_row(row):
    hmm = re.finditer(r"([O.]+)", row)
    s = 0
    l = len(row)
    out = np.array(list(row.replace("O", ".")))
    for m in hmm:
        n = m.group(1).count("O")
        if n:
            s += sum(range(l - m.start(), l - m.start() - n, -1))
            out[m.start():m.start() + n] = "O"
    return s, "".join(out)

def a(data, rotations):
    grid = text_to_numpy(data)
    total = 0
    curr_grid = grid
    for rotation in range(rotations):
        new_rows = []
        for row in np.rot90(curr_grid):
            s, new_row = compute_row(row.tostring().decode())
            total += s
            new_rows.append(new_row)
        curr_grid = text_to_numpy("\n".join(new_rows))
    return total


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data, 1)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data, 1)
print("a:", answer)
assert answer == 108759


# Part b
for example in puzzle.examples:
    if example.answer_b:
        example_answer = a(example.input_data, 1000000000*4)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = a(puzzle.input_data, 1000000000*4)
print("b:", answer)
puzzle.answer_b = answer
