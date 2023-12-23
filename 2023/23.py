import datetime
import re
import queue
from collections import defaultdict
import sys

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


sys.setrecursionlimit(10000)


# Part a
def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from.pop(current)
        total_path.append(current)
    return total_path


def debug_print(forest, path):
    height, width = int(forest[-1].real + 1), int(forest[-1].imag + 1)
    for i in range(height):
        for j in range(width):
            if complex(i, j) in forest:
                print("#", end="")
            elif complex(i, j) in path:
                print("O", end="")
            else:
                print(".", end="")
        print("")


def b_star(paths, end, forest, slopes):
    if sum(p[-1] == end for p in paths) == len(paths):
        return paths
    new_paths = []
    for path in paths:
        current = path[-1]
        if current == end:
            new_paths.append(path)
            continue
        dirs = [0 + 1j, -1 + 0j, 0 - 1j, 1 + 0j]
        for dir in dirs:
            next_node = current + dir
            if next_node in forest:
                continue
            if next_node in path:
                continue
            if next_node in slopes:
                if dir != slopes[next_node]:
                    continue
            new_paths.append([*path, next_node])
    return b_star(new_paths, end, forest, slopes)


def a(data):
    forest = []
    slopes = {}
    dirs = {
        "^": -1 + 0j,
        ">": 0 + 1j,
        "v": 1 + 0j,
        "<": 0 - 1j,
    }
    for i, line in enumerate(data.splitlines()):
        for m in re.finditer("#", line):
            j = m.start()
            forest.append(complex(i, j))
        for m in re.finditer("\^|>|v|<", line):
            j = m.start()
            slopes[complex(i, j)] = dirs[m.group(0)]
    height, width = i + 1, j + 1

    start = 0 + 1j
    dir = 1 + 0j

    goal = complex(height - 1, width - 2)

    forest.append(start - 1)
    forest.append(goal + 1)

    paths =  b_star([[start]], goal, forest, slopes)
    return max(len(p) for p in paths) - 1


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {94})")
        assert example_answer == 94
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
