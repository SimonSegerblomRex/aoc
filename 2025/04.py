import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    pos = []
    for j, line in enumerate(data.split()):
        for i, ch in enumerate(line):
            if ch == "@":
                pos.append(i + j*1j)
    c = 0
    for p in pos:
        neighbours = {p + 1, p + 1 - 1j, p - 1j, p - 1 - 1j, p - 1, p - 1 + 1j, p + 1j, p + 1 + 1j}
        if len(neighbours & set(pos)) < 4:
            c += 1
    return c


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
