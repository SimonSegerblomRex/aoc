import datetime
import re
from collections import Counter
from functools import cache

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)

example = "0 1 10 99 999"


def blinkhelper(stone):
    ss = str(stone)
    if stone == 0:
        return [1]
    elif not len(ss) % 2:
        w = len(ss)
        return [int(ss[:w//2]), int(ss[w//2:])]
    return [2024 * stone]


@cache
def blinkk(stones, blinks):
    if blinks == 0:
        return stones
    new = []
    for s in stones:
        new += blinkhelper(s)
    new = sorted(new)
    return blinkk(tuple(new), blinks - 1)


@cache
def blink(stone, blinks):
    return blinkk((stone,), blinks)


# Part a
def a(data, blinks=25):
    stones = list(map(int, data.split()))
    chunk = 25
    tot = 0
    for _ in range(blinks // chunk):
        stones = Counter(stones)
        new = []
        for s, c in stones.items():
            new += c * blink(s, chunk)
        stones = new
    return sum(Counter(stones).values())


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
