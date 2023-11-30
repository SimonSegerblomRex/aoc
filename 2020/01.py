import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = 2020
DAY = 1

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    data = list(map(int, data.splitlines()))
    for e1 in data:
        for e2 in data:
            if e1 + e2 == 2020:
                return e1 * e2
    raise


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
    data = list(map(int, data.splitlines()))
    for e1 in data:
        for e2 in data:
            if e1 + e2 > 2020:
                continue
            for e3 in data:
                if e1 + e2 + e3 == 2020:
                    return e1 * e2 * e3


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
