import datetime
import re

# from itertools import batched

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    s = 0
    for line in data.split(","):
        lo, hi = line.split("-")
        for n in range(int(lo), int(hi) + 1):
            n = str(n)
            if len(n) % 2:
                continue
            if n[: len(n) // 2] == n[len(n) // 2 :]:
                s += int(n)
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 44854383294


def batched(s, n):
    return (s[i * n : i * n + n] for i in range(len(s) // n))


# Part b
def b(data):
    s = 0
    for line in data.split(","):
        lo, hi = line.split("-")
        for n in range(int(lo), int(hi) + 1):
            n = str(n)
            invalid = False
            for c in range(1, len(n) // 2 + 1):
                if len(n) % c:
                    continue
                if len(set(batched(n, c))) == 1:
                    invalid = True
                if invalid:
                    break
            if invalid:
                s += int(n)
    return s


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
