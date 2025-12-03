import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    s = 0
    for line in data.split():
        d1 = "0"
        d2 = "0"
        for i, c in enumerate(line[:-1]):
            if c > d1:
                d1 = c
                d2 = line[i + 1]
                continue
            if c > d2:
                d2 = c
        if line[-1] > d2:
            d2 = line[-1]
        print(d1+ d2)
        s += int(d1 + d2)
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
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
