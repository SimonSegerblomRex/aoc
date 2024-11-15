from itertools import islice

import numpy as np
from aocd.models import Puzzle

YEAR = 2019
DAY = 16

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, phases=100):
    inp = list(map(int, data))
    sz = len(inp)
    pattern = [0, 1, 0, -1]
    out = [0] * sz
    for p in range(phases):
        for e in range(1 , sz + 1):
            s = 0
            for i, d in enumerate(inp[e - 1:], e):
                idx = (i % (4 * e)) // e
                s += d * pattern[idx]
            out[e - 1] = abs(s) % 10
        inp = out
    return int("".join(str(n) for n in inp[:8]))


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 61149209


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
