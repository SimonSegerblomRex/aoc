import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a


def get_new_beams(pos, dir, kind):
    if kind == ".":
        return [(pos + dir, dir)]
    if kind == "|":
        if dir in (0 + 1j, 0 - 1j):
            return [(pos + 1, 1 + 0j), (pos - 1, -1 + 0j)]
        return [(pos + dir, dir)]
    if kind == "-":
        if dir in (1 + 0j, -1 + 0j):
            return [(pos - 1j, 0 - 1j), (pos + 1j, 0 + 1j)]
        return [(pos + dir, dir)]
    if kind == "/":
        new_dir = complex(-dir.imag, -dir.real)
        return [(pos + new_dir, new_dir)]
    if kind == "\\":
        new_dir = complex(dir.imag, dir.real)
        return [(pos + new_dir, new_dir)]
    raise ValueError


def a(data):
    grid = {}
    dirs = (0 + 1j, -1 + 0j, 0 - 1j, 1 + 0j)
    rows = data.splitlines()
    height, width = len(rows), len(rows[0])
    for i, row in enumerate(rows):
        for j, c in enumerate(row):
            pos = complex(i, j)
            for dir in dirs:
                grid[(pos, dir)] = get_new_beams(pos, dir, c)
    # Add border
    if 0:
        for i in range(-1, height + 1):
            for j in (-1, width):
                pos = complex(i, j)
                for dir in dirs:
                    grid[pos, dir] = []
        for i in (-1, height):
            for j in range(-1, width + 1):
                pos = complex(i, j)
                for dir in dirs:
                    grid[pos, dir] = []
    beams = set([(0 + 0j, 0 + 1j)])
    visited = set()
    while beams:
        pos, dir = beams.pop()
        if pos.real < 0 or pos.real > height - 1 or pos.imag < 0 or pos.imag > width - 1:
            continue
        visited.add((pos, dir))
        new_beams = grid[(pos, dir)]
        beams.update(new_beams)
        beams.difference_update(visited)
    # Debug print
    if 0:
        for i in range(height):
            print("")
            for j in range(width):
                if complex(i, j) in visited:
                    print("#", end="")
                else:
                    print(".", end="")
    return len(set(p for p, _ in visited))


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
