from functools import cache

import numpy as np
from sympy import Abs, I, im, re
from aocd.models import Puzzle

YEAR = 2019
DAY = 10

puzzle = Puzzle(year=YEAR, day=DAY)

@cache
def norm(x):
    tmp = x / Abs(x)
    # sympy doesn't simplify the fractions
    # if we don't do anything special...
    # using re + im seems to be the
    # cheapest way of achieving this
    return re(tmp), im(tmp)


# Part a
def a(data):
    asteroids = []
    for i, line in enumerate(data.splitlines()):
        for j, c in enumerate(line):
            if c == "#":
                asteroids.append(i + j*I)
    best = {}
    for a in asteroids:
        tmp = set()
        for b in asteroids:
            if a == b:
                continue
            tmp.add(norm(b - a))
        best = tmp if len(tmp) > len(best) else best
    return len(best)


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 309


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
