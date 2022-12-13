import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


def check(l0, l1):
    if not l0 and not l1:
        return None
    if not l0:
        return True
    if not l1:
        return False
    for v0, v1 in zip(l0, l1):
        if isinstance(v0, int) and isinstance(v1, int):
            if v0 < v1:
                return True
            if v1 < v0:
                return False
        elif isinstance(v0, list) and isinstance(v1, list):
            c = check(v0, v1)
            #return c
            if c is not None:
                return c
        elif isinstance(v0, list) and isinstance(v1, int):
            c = check(v0, [v1])
            #return c
            if c is not None:
                return c
        elif isinstance(v0, int) and isinstance(v1, list):
            c = check([v0], v1)
            #return c
            if c is not None:
                return c
        else:
            print("Shouldn't end up here!")
            breakpoint()
    if len(l1) < len(l0):
        return False
    if len(l0) < len(l1):
        return True
    return None

# Part a
def a(data):
    pairs = []
    for lines in data.split("\n\n"):
        pairs.append([eval(line) for line in lines.split("\n")])
    s = 0
    for i, p in enumerate(pairs):
        c = check(p[0], p[1])
        if c or c is None:
            print("r:", i + 1)
            s += i + 1
    return s

example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 13
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 6428


# Part b
def b(data):
    exit()

example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == ...
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
