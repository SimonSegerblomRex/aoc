import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    #data.spli
    rows = [np.frombuffer(row.encode(), dtype=np.uint8) for row in data.splitlines()]
    grid = np.vstack(rows)
    grid = np.pad(grid, ord("."))
    start_pos = np.where(grid == ord("S"))
    start_pos = (start_pos[0][0], start_pos[1][0])
    s = 0
    pos = start_pos
    prev_pos = (-1, -1)
    prev_pipe = ord("S")
    while True:
        s += 1
        i, j = pos
        if grid[i, j + 1] in [ord("-"), ord("J"), ord("7"), ord("S")] and prev_pos != (i, j + 1) and grid[i, j] in [ord("-"), ord("L"), ord("F"), ord("S")]:
            j = j + 1
        elif grid[i - 1, j] in [ord("|"), ord("7"), ord("F"), ord("S")] and prev_pos != (i- 1, j) and grid[i, j] in [ord("|"), ord("L"), ord("J"), ord("S")]:
            i = i - 1
        elif grid[i, j - 1] in [ord("-"), ord("L"), ord("F"), ord("S")] and prev_pos != (i, j - 1) and grid[i, j] in [ord("-"), ord("J"), ord("7"), ord("S")]:
            j = j - 1
        elif grid[i + 1, j] in [ord("|"), ord("L"), ord("J"), ord("S")] and prev_pos != (i + 1, j)  and grid[i, j] in [ord("|"), ord("F"), ord("7"), ord("S")]:
            i = i + 1
        else:
            print(grid[i-1:i+2, j-1:j+2])
            breakpoint()
        prev_pos = pos
        prev_pipe = grid[prev_pos]
        pos = (i, j)
        if pos == start_pos:
            break
    return s // 2


example = """.....
.S-7.
.|.|.
.L-J.
....."""
example_answer =  a(example)
assert example_answer == 4

print("OKK")

example = """..F7.
.FJ|.
SJ.L7
|F--J
LJ..."""
example_answer =  a(example)
assert example_answer == 8

print("AAA")
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
