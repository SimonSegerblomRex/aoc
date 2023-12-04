import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    lines = iter(data.splitlines())
    p = 0
    for n, line in enumerate(lines, 1):
        l1, l2 = line.split("|")
        l1 = l1.split(":")[1]
        l1 = [int(e) for e in l1.strip().split(" ") if e != ""]
        l2 = [int(e) for e in l2.strip().split(" ") if e != ""]
        tmp = list(set(l2).intersection(l1))
        if tmp:
            p += 2**(len(tmp) - 1)
    return p


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
#puzzle.answer_a = answer


# Part b
def b(data):
    lines = data.splitlines()
    copies = [0] * (len(lines)  + 1)
    p = len(lines)
    for n, line in enumerate(lines, 1):
        l1, l2 = line.split("|")
        l1 = l1.split(":")[1]
        l1 = [int(e) for e in l1.strip().split(" ") if e != ""]
        l2 = [int(e) for e in l2.strip().split(" ") if e != ""]
        tmp = list(set(l2).intersection(l1))
        #breakpoint()
        if tmp:
            for i in range(n + 1, n + len(tmp) + 1):
                copies[i] += 1 + copies[n]
            print(copies)
    return p + sum(copies)
    breakpoint()
    return p


example = """Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53
Card 2: 13 32 20 16 61 | 61 30 68 82 17 32 24 19
Card 3:  1 21 53 59 44 | 69 82 63 72 16 21 14  1
Card 4: 41 92 73 84 69 | 59 84 76 51 58  5 54 83
Card 5: 87 83 26 28 32 | 88 30 70 12 93 22 82 36
Card 6: 31 18 13 56 72 | 74 77 10 23 35 67 36 1"""

print(b(example))
assert b(example) == 30

answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
