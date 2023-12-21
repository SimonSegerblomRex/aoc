import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, max_steps):
    rocks = []
    start = None
    for i, line in enumerate(data.splitlines()):
        for m in re.finditer("#", line):
            j = m.start()
            rocks.append(complex(i, j))
        if start is None:
            if (m := re.search("S", line)) is not None:
                start = complex(i, m.start())
    curr_pos = set([(start, 0)])
    #visited = set()
    visited = {}
    dirs = [0 + 1j, -1 + 0j, 0 - 1j, 1 + 0j]
    while curr_pos:
        pos, steps = curr_pos.pop()
        if steps > max_steps:
            continue
        for dir in dirs:
            next_pos = pos + dir
            if next_pos in rocks:
                continue
            if next_pos in visited:
                if steps >= visited[next_pos]:
                    continue
            curr_pos.add((next_pos, steps + 1))
        if pos not in visited:
            visited[pos] = steps
        else:
            visited[pos] = min(steps, visited[pos])
        print(len(visited), len(curr_pos), steps)
    visited = set([pos for pos, s in visited.items() if not s % 2])
    if 0:
        # Debug print
        lines = data.splitlines()
        height, width = len(lines), len(lines[0])

        for i in range(height):
            for j in range(width):
                if complex(i, j) in rocks:
                    print("#", end="")
                elif complex(i, j) in visited:
                    print("O", end="")
                else:
                    print(".", end="")
            print("")
        breakpoint()
    return len(visited)


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data, max_steps=6)
        print(f"Example answer: {example_answer} (expecting: {16})")
        assert example_answer == 16
answer = a(puzzle.input_data, max_steps=64)
print("a:", answer)
assert answer > 3459
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
