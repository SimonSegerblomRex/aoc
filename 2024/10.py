from collections import defaultdict

import numpy as np
from aocd.models import Puzzle

YEAR = 2024
DAY = 10

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    points = defaultdict(list)
    grid = {}
    for i, line in enumerate(data.splitlines()):
        width = len(line)
        for j, c in enumerate(line):
            points[c].append(j + i*1j)
            grid[j + i*1j] = int(c)
    s = 0
    def neighbours(p):
        return p + 1, p - 1j, p - 1, p + 1j
    for pos in points["0"]:
        curr = [pos]
        while curr:
            new = []
            score = set()
            for p in curr:
                if grid[p] == 9:
                    score.add(p)
                neigh = neighbours(p)
                neigh = [n for n in neigh if n in grid]
                neigh = [n for n in neigh if grid[n] - 1 == grid[p]]
                new.extend(neigh)
            curr = new
        s += len(score)
    return s


example = """89010123
78121874
87430965
96549874
45678903
32019012
01329801
10456732"""
answer = a(example)
assert answer == 36
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 514


# Part b
def b(data):
    points = defaultdict(list)
    grid = {}
    for i, line in enumerate(data.splitlines()):
        width = len(line)
        for j, c in enumerate(line):
            points[c].append(j + i*1j)
            grid[j + i*1j] = int(c)
    s = 0
    def neighbours(p):
        return p + 1, p - 1j, p - 1, p + 1j
    for pos in points["0"]:
        curr = [pos]
        while curr:
            new = []
            for p in curr:
                if grid[p] == 9:
                    s += 1
                neigh = neighbours(p)
                neigh = [n for n in neigh if n in grid]
                neigh = [n for n in neigh if grid[n] - 1 == grid[p]]
                new.extend(neigh)
            curr = new
    return s


answer = b(example)
assert answer == 81
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 1162
