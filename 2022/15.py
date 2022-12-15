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
#answer = a(puzzle.input_data, y=2000000)
#print("a:", answer)
#assert answer == 5367037


# Part b
def find_x(data, y):
    x = set()
    exclude = set()
    for xs, ys, xb, yb in data:
        md = np.abs(xs - xb) + np.abs(ys - yb)
        if (yd := np.abs(ys - y)) <= md:
            x.update(range(xs - (md - yd), xs + (md - yd) + 1))
        #qif yb == y:
        #    exclude.add(xb)
    return list(x)


def b(data, max_c):
    data = re.findall(PATTERN, data)
    data = [tuple(map(int, e)) for e in data]
    grid = np.zeros((max_c + 1, max_c + 1), dtype=bool)
    for i in range(grid.shape[0]):
        xc = np.array(find_x(data, i))
        xc = xc[(xc >=0 ) & (xc <= max_c)]
        grid[i, xc] = True
    y, x = np.argwhere(grid == False).tolist()[0]
    return x * 4000000 + y


example_answer = b(puzzle.example_data, max_c=20)
print(example_answer)
assert example_answer == 56000011
answer = b(puzzle.input_data, max_c=4000000)
print("b:", answer)
puzzle.answer_b = answer
