import datetime
import re
from functools import cache

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


numeric_keypad = {
    7: 0j,
    8: 1 + 0j,
    9: 2 + 0j,
    4: 0 + 1j,
    5: 1 + 1j,
    6: 2 + 1j,
    1: 0 + 2j,
    2: 1 + 2j,
    3: 2 + 2j,
    0: 1 + 3j,
    10: 2 + 3j,  # A
}


directional_keypad = {
    "^": 1 + 0j,
    "A": 2 + 0j,
    "<": 0 + 1j,
    "v": 1 + 1j,
    ">": 2 + 1j,
}


@cache
def path_numeric(start, goal):
    s = numeric_keypad[start]
    g = numeric_keypad[goal]
    path = ""
    dx = int(g.real - s.real)
    dy = int(g.imag - s.imag)
    dy = -dy
    if start in [0, 10] and goal in [1, 4, 7]:
        path = "^" * dy
        dy = 0
    if start in [1, 7, 4] and goal in [0, 10]:
        path = ">" * dx
        dx = 0
    if dx < 0:
        path += "<" * -dx
    if dy < 0:
        path += "v" * -dy
    if dy > 0:
        path += "^" * dy
    if dx > 0:
        path += ">" * dx
    return path


@cache
def path_directional(start, goal):
    s = directional_keypad[start]
    g = directional_keypad[goal]
    path = ""
    dx = int(g.real - s.real)
    dy = int(g.imag - s.imag)
    dy = -dy
    if start in ["^", "A"] and goal in ["<"]:
        path = "v"
        dy = 0
    if dx < 0:
        path += "<" * -dx
    if dy < 0:
        path += "v" * -dy
    if dx > 0:
        path += ">" * dx
    if dy > 0:
        path += "^" * dy
    return path


# Part a
def a(data):
    codes = data.split()
    s = 0
    pos_d1 = "A"
    pos_d2 = "A"
    pos_n = 10
    for code_str in codes:
        code = [int(n, 16) for n in code_str]
        c = 0
        for target in code:
            path_n = path_numeric(pos_n, target)
            # Move to target digit on numerical pad
            for d in path_n:
                path_d2 = path_directional(pos_d2, d)
                pos_d2 = d
                for k in path_d2:
                    path_d1 = path_directional(pos_d1, k)
                    pos_d1 = k
                    c += len(path_d1)
                    c += 1  # for A
                path_d1 = path_directional(pos_d1, "A")
                pos_d1 = "A"
                c += len(path_d1)
                c += 1  # for A
            # Time to press A on directional pad 2
            path_d2 = path_directional(pos_d2, "A")
            pos_d2 = "A"
            for k in path_d2:
                path_d1 = path_directional(pos_d1, k)
                pos_d1 = k
                c += len(path_d1)
                c += 1  # for A
            path_d1 = path_directional(pos_d1, "A")
            pos_d1 = "A"
            c += len(path_d1)
            c += 1  # for A
            pos_n = target
        s += int(code_str[:3]) * c
    return s


example = """029A
980A
179A
456A
379A"""
example_answer = a(example)
print(example_answer)
assert example_answer == 126384
answer = a(puzzle.input_data)
print("a:", answer)
assert answer < 163872
assert answer > 160060
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
