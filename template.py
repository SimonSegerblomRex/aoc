import datetime

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)

# Part a
def a(input, debug=False):
    if debug:
        print(input)
        breakpoint()
    #input = np.fromstring(input, dtype=int, sep="\n"
    #output = ...
    if debug:
        print(output)
        breakpoint()
    return output

ANSWER_EXAMPLE = None
try:
    assert a(puzzle.example_data) == ANSWER_EXAMPLE
except Exception as e:
    print(e)
    assert a(puzzle.example_data, debug=True) == ANSWER_EXAMPLE
try:
    answer = a(puzzle.input_data)
except Exception as e:
    print(e)
    answer = a(puzzle.input_data, debug=True)
puzzle.answer_a = answer

# Part b
def b(input, debug=False):
    if debug:
        print(input)
        breakpoint()
    #input = np.fromstring(input, dtype=int, sep="\n"
    #output = ...
    if debug:
        print(output)
        breakpoint()
    return output

ANSWER_EXAMPLE = None
try:
    assert b(puzzle.example_data) == ANSWER_EXAMPLE
except Exception as e:
    print(e)
    assert b(puzzle.example_data, debug=True) == ANSWER_EXAMPLE
try:
    answer = b(puzzle.input_data)
except Exception as e:
    print(e)
    answer = b(puzzle.input_data, debug=True)
puzzle.answer_b = answer
