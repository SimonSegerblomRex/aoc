import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    problems = []
    for j, line in enumerate(data.split("\n")):
        for i, e in enumerate([e for e in  line.split(" ") if e]):
            if i > len(problems) - 1:
                problems.append([])
            problems[i].append(e)
    s = 0
    for problem in problems:
        if problem[-1] == "+":
            tmp = 0
        else:
            tmp = 1
        for e in problem[:-1]:
            if problem[-1] == "+":
                tmp += int(e)
            else:
                tmp *= int(e)
        s += tmp
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 5335495999141


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
