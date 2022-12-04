import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 6

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, days):
    data = np.fromstring(data, dtype=np.uint8, sep=",")
    total = 0
    for fish in data:
        old_fish = np.bincount([fish], minlength=7)
        new_fish = np.zeros(9, dtype=int)
        for day in range(1, days + 1):
            old_zero = old_fish[0]
            old_fish = np.roll(old_fish, -1)
            new_zero = new_fish[0]
            new_fish = np.roll(new_fish, -1)
            old_fish[6] = old_zero + new_zero
            new_fish[8] = old_zero + new_zero
        total += np.sum(old_fish) + np.sum(new_fish)
    return total

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
