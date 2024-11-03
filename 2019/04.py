import numpy as np
from aocd.models import Puzzle
from more_itertools import pairwise

YEAR = 2019
DAY = 4

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    min_lim, max_lim = map(int, data.split("-"))
    s = 0
    for n in range(min_lim, max_lim + 1):
        digits = list(map(int, (str(n))))
        double = False
        increasing = True
        for d0, d1 in pairwise(digits):
            if d1 < d0:
                increasing = False
                break
            if d0 == d1:
                double = True
        if double and increasing:
            s += 1
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 979


# Part b
def b(data):
    min_lim, max_lim = map(int, data.split("-"))
    s = 0
    for n in range(min_lim, max_lim + 1):
        digits = list(map(int, (str(n))))
        doubles = []
        increasing = True
        for i, (d0, d1) in enumerate(pairwise(digits)):
            if d1 < d0:
                increasing = False
                break
            if d0 == d1:
                doubles.append((i, d0))
        if doubles and increasing:
            vals = set(v for _, v in doubles)
            counts = []
            for val in vals:
                counts.append(sum(v == val for _, v in doubles))
            if min(counts) > 1:
                continue
            s += 1
    return s


answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 635
