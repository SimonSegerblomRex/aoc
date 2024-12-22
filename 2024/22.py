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
    for n in data.split():
        for _ in range(2000):
            n = int(n)
            n ^= 64 * n
            n %= 16777216
            n ^= n // 32
            n %= 16777216
            n ^= 2048 * n
            n %= 16777216
        s += n
    return s



example = """1
10
100
2024"""
answer = a(example)
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 12664695565


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
