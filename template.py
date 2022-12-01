import datetime

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(input):
    ...

ANSWER_EXAMPLE = ...
assert a(puzzle.example_data) == ANSWER_EXAMPLE
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(input):
    ...

ANSWER_EXAMPLE = ...
assert b(puzzle.example_data) == ANSWER_EXAMPLE
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
