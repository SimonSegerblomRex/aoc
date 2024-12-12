import datetime
import re
from collections import defaultdict

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, sides=False):
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
    if not sides:
        s = 0
        for r, b in zip(regions, borders):
            s += len(r) * len(b)
        return s
    sides = []

    for border in borders:
        lines = []
        border = sorted(border, key=lambda x: (x[0].real, x[1].real, x[0].imag, x[1].imag))
        for p, dir in border:
            for line in lines:
                if dir.real == 0:
                    if ((p + 1, dir) in line) or ((p - 1, dir) in line):
                        line.append((p, dir))
                        break
                elif dir.imag == 0:
                    if ((p + 1j, dir) in line) or ((p - 1j, dir) in line):
                        line.append((p, dir))
                        break
            else:
                lines.append([(p, dir)])
        sides.append(lines)

    s = 0
    for r, t in zip(regions, sides):
        s += len(r) * len(t)
    return s


if 0:
    answer = a(puzzle.input_data)
    print("a:", answer)
    puzzle.answer_a = answer


# Part b
example = """AAAA
BBCD
BBCC
EEEC"""
answer = a(example, sides=True)
print(answer)
answer = a(puzzle.input_data, sides=True)
print("b:", answer)
puzzle.answer_b = answer
