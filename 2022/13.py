import datetime
import functools
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


def check(l0, l1):
    if not l0 and not l1:
        return 0
    if not l0:
        return 1
    if not l1:
        return -1
    for v0, v1 in zip(l0, l1):
        if isinstance(v0, int) and isinstance(v1, int):
            if v0 < v1:
                return 1
            if v1 < v0:
                return -1
        elif isinstance(v0, list) and isinstance(v1, list):
            c = check(v0, v1)
            #return c
            if c != 0:
                return c
        elif isinstance(v0, list) and isinstance(v1, int):
            c = check(v0, [v1])
            #return c
            if c != 0:
                return c
        elif isinstance(v0, int) and isinstance(v1, list):
            c = check([v0], v1)
            #return c
            if c != 0:
                return c
        else:
            print("Shouldn't end up here!")
            breakpoint()
    if len(l1) < len(l0):
        return -1
    if len(l0) < len(l1):
        return 1
    return 0

# Part a
def a(data):
    pairs = []
    for lines in data.split("\n\n"):
        pairs.append([eval(line) for line in lines.split("\n")])
    s = 0
    for i, p in enumerate(pairs):
        if check(p[0], p[1]) >= 0:
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
    packets = [eval(line) for line in data.replace("\n\n", "\n").splitlines()]
    packets.append([[2]])
    packets.append([[6]])
    packets = sorted(packets, key=functools.cmp_to_key(check), reverse=True)
    i0 = None
    i1 = None
    for i, p in enumerate(packets):
        if p == [[2]]:
            i0 = i + 1
        elif p == [[6]]:
            i1 = i + 1
    return i0 * i1

example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 140
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
