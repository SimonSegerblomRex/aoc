import numpy as np
from aocd.models import Puzzle
from more_itertools import pairwise

YEAR = 2019
DAY = 4

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    min_lim, max_lim = map(int, data.split("-"))
    s = 0
    for n in range(min_lim, max_lim + 1):
        digits = list(map(int, (str(n))))
        double = False
        increasing = True
        for d0, d1 in pairwise(digits):
            if d1 < d0:
                increasing = False
                break
            if d0 == d1:
                double = True
        if double and increasing:
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
