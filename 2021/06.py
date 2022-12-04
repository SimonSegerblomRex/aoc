import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 6

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, days):
    data = np.fromstring(data, dtype=np.uint8, sep=",")
    count = np.bincount(data, minlength=9)
    for day in range(1, days + 1):
        zero_count = count[0]
        count = np.roll(count, -1)
        count[6] += zero_count
        count[8] = zero_count
    return np.sum(count)

answer_example = a(puzzle.example_data, days=80)
print("example a:", answer_example)
assert answer_example == 5934
answer = a(puzzle.input_data, days=80)
print("a:", answer)
assert answer == 362666


# Part b
answer_example = a(puzzle.example_data, days=256)
print("example b:", answer_example)
assert answer_example == 26984457539
answer = a(puzzle.input_data, days=256)
print("b:", answer)
assert answer == 1640526601595
