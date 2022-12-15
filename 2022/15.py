import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)

PATTERN = r"Sensor at x=([-\d]+), y=([-\d]+): closest beacon is at x=([-\d]+), y=([-\d]+)"

# Part a
def a(data, y):
    data = re.findall(PATTERN, data)
    data = [tuple(map(int, e)) for e in data]
    x = set()
    exclude = set()
    for xs, ys, xb, yb in data:
        md = np.abs(xs - xb) + np.abs(ys - yb)
        if (yd := np.abs(ys - y)) <= md:
            x.update(range(xs - (md - yd), xs + (md - yd) + 1))
        if yb == y:
            exclude.add(xb)

    return len(x - exclude)


example_answer = a(puzzle.example_data, y=10)
print(example_answer)
assert example_answer == 26
answer = a(puzzle.input_data, y=2000000)
print("a:", answer)
puzzle.answer_a = answer
####B######################

# Part b
def b(data):
    exit()


example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == ...
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
