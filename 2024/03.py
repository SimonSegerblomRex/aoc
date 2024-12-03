import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    tmp = re.findall("mul\((\d+),(\d+)\)", data)
    s = 0
    for t in tmp:
        s += int(t[0]) * int(t[1])
    return s


answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    pause = False
    s = 0
    for m in re.finditer("mul\((\d+),(\d+)\)|(do\(\))|(don't\(\))", data):
        if m.group(0) == "don't()":
            pause = True
            continue
        if m.group(0) == "do()":
            pause = False
            continue
        if pause:
            continue
        s += int(m.group(1)) * int(m.group(2))
    return s


answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
