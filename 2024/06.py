import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    obstacles = []
    for i, line in enumerate(data.splitlines()):
        width = len(line)
        for j, c in enumerate(line):
            if c == "#":
                obstacles.append(j + i*1j)
            if c == "^":
                pos = j + i*1j
    height = i + 1
    dir = -1j
    c = 0
    visited = set()
    while (0 <= pos.real <= width) and (0 <= pos.imag <= height):
        visited.add(pos)
        if pos + dir in obstacles:
            dir *= 1j
            continue
        pos += dir
        c += 1
    return len(visited)
    breakpoint()

example = """....#.....
.........#
..........
..#.......
.......#..
..........
.#..^.....
........#.
#.........
......#..."""
answer = a(example)
#assert answer == 41
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
