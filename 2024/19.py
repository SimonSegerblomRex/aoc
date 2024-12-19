import datetime
import re
from functools import cache

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)



# Part a
def a(data):
    patterns, designs = data.split("\n\n")
    patterns = patterns.split(", ")
    designs = designs.split()
    @cache
    def check(design):
        if not design or design in patterns:
            return True
        ok = False
        for i in range(len(design)):
            if design[:i] in patterns:
                ok = check(design[i:])
                if ok:
                    break
        return ok
    s = 0
    for design in designs:
        if check(design):
            s += 1
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
