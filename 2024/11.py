import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)

example = "0 1 10 99 999"

# Part a
def a(data, blinks=25):
    stones = list(map(int, data.split()))
    for _ in range(blinks):
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
        stones = new
    return len(stones)


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
