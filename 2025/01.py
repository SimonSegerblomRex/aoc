import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = 2025
DAY = 1

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    c = 0
    l = 50
    for i in data.split():
        if i[0] == "L":
            l -= int(i[1:])
        else:
            l += int(i[1:])
        l %= 100
        if l == 0:
            c += 1
    return c


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 1078

# Part b
def b(data):
    c = 0
    l = 50
    for i in data.split():
        for t in range(int(i[1:])):
            if i[0] == "L":
                l -= 1
            else:
                l += 1
            l %= 100
            if l == 0:
                c += 1
    return c

answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
