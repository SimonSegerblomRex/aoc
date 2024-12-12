import datetime
import re
from collections import defaultdict

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    plants = {}
    for i, line in enumerate(data.splitlines()):
        width = len(line)
        for j, c in enumerate(line):
            plants[j + i*1j] = c
    height = i + 1
    visited = []
    regions = []
    def neighbours(p):
        return [p + 1, p - 1j, p - 1, p + 1j]
    for i in range(height):
        for j in range(width):
            pos = j + i*1j
            if pos in visited:
                continue
            region = [pos]
            keep_going = True
            while keep_going:
                keep_going = False
                for p in region:
                    neigh = neighbours(p)
                    for n in neigh:
                        if n not in region:
                            if (0 <= n.real < width) and (0 <= n.imag < height) and (plants[n] == plants[p]):
                                region.append(n)
                                keep_going = True
            visited += region
            regions.append(region)
            # TODO: add region to visited
    def neighboursd(p):
        return [p + 1, p - 1j, p - 1, p + 1j, p + 1 - 1j, p + 1 + 1j, p - 1 + 1j, p - 1 - 1j]
    borders = []
    for region in regions:
        border = set()
        for p in region:
            neigh = neighbours(p)
            for n in neigh:
                if (not (0 <= n.real < width)) or (not (0 <= n.imag < height)) or (plants[n] != plants[p]):
                    border.add((n, n - p))
        borders.append(border)
    s = 0
    for r, b in zip(regions, borders):
        s += len(r) * len(b)
    return s



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
