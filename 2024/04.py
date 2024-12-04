import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    rows = np.array([[ord(c) for c in line] for line in data.splitlines()], dtype=int)
    s = 0
    for x in rows, np.rot90(rows):
        for line in x:
            line = "".join(chr(n) for n in line)
            s += line.count("XMAS")
            s += line.count("SAMX")
        for i in range(-rows.shape[0] + 1, rows.shape[0]):
            line = np.diag(x, i)
            line = "".join(chr(n) for n in line)
            s += line.count("XMAS")
            s += line.count("SAMX")
    return s


answer = a(puzzle.input_data)
print("a:", answer)
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
