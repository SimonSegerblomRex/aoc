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
breakpoint()
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
