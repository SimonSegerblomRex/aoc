import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)

example = "0 1 10 99 999"


def blink(stones):
    new = []
    for s in stones:
        ss = str(s)
        if s == 0:
            new.append(1)
        elif not len(ss) % 2:
            w = len(ss)
            new.append(int(ss[:w//2]))
            new.append(int(ss[w//2:]))
        else:
            new.append(2024 * s)
    return new


def blink2(stones, magic):
    new = []
    extra = 0
    for s in stones:
        ss = str(s)
        if s == 0:
            extra += magic
        elif not len(ss) % 2:
            w = len(ss)
            new.append(int(ss[:w//2]))
            new.append(int(ss[w//2:]))
        else:
            new.append(2024 * s)
    return new, extra


# Part a
def a(data, blinks=25):
    stones = [0]
    zero = []
    for _ in range(blinks):
        stones = blink(stones)
        zero.append(len(stones))
    stones = list(map(int, data.split()))
    extra = 0
    for i, _ in enumerate(range(blinks)):
        stones, plus = blink2(stones, zero[blinks - 1 - i])
        extra += plus
    return len(stones) + extra


example = "125 17"
answer = a(example)
print(answer)
assert answer == 55312
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
answer = a(puzzle.input_data, blinks=75)
print("b:", answer)
puzzle.answer_b = answer
