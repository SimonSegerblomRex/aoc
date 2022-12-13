import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


def check(l0, l1):
    #if len(l1) < len(l0):
    #    return False
    #l1 = l1[:len(l0)]
    """
    if not l0:
        return True
    if not l1:
        return False
    """
    for v0, v1 in zip(l0, l1):
        if isinstance(v0, int) and isinstance(v1, int):
            if v0 < v1:
                return True
            elif v1 < v0:
                return False
        elif isinstance(v0, list) and isinstance(v1, list):
            return check(v0, v1)
        elif isinstance(v0, list) and isinstance(v1, int):
            return check(v0, [v1])
        elif isinstance(v0, int) and isinstance(v1, list):
            return check([v0], v1)
        else:
            print("Shouldn't end up here!")
            breakpoint()
    if len(l1) < len(l0):
        return False
    return True

# Part a
def a(data):
    pairs = []
    for lines in data.split("\n\n"):
        pairs.append([eval(line) for line in lines.split("\n")])
    s = 0
    for i, p in enumerate(pairs):
        if check(p[0], p[1]):
            print(i + 1)
            s += i + 1
    return s

example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 13
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    exit()

example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == ...
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
