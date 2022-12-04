import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 6

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a_old(data, days):
    data = np.fromstring(data, dtype=np.uint8, sep=",")
    old_fish = data
    new_fish = np.empty(0, dtype=np.uint8)
    for day in range(1, days + 1):
        old_zero = np.flatnonzero(old_fish == 0)
        old_fish -= 1
        new_zero = np.flatnonzero(new_fish == 0)
        new_fish -= 1
        old_fish[old_zero] = 6
        old_fish = np.append(
            old_fish,
            np.full(shape=len(new_zero), fill_value=6),
        )
        new_fish = np.delete(new_fish, new_zero)
        new_fish = np.append(
            new_fish,
            np.full(shape=len(old_zero) + len(new_zero), fill_value=8),
        )
        if day < 19:
            print("Day", day, old_fish, new_fish)
    return len(old_fish) + len(new_fish)

def a_old2(data, days):
    data = np.fromstring(data, dtype=np.uint8, sep=",")
    total = 0
    for fish in data:
        old_fish = np.array([fish], dtype=np.uint8)
        new_fish = np.empty(0, dtype=np.uint8)
        for day in range(1, days + 1):
            old_zero = np.flatnonzero(old_fish == 0)
            old_fish -= 1
            new_zero = np.flatnonzero(new_fish == 0)
            new_fish -= 1
            old_fish[old_zero] = 6
            old_fish = np.append(
                old_fish,
                np.full(shape=len(new_zero), fill_value=6),
            )
            new_fish = np.delete(new_fish, new_zero)
            new_fish = np.append(
                new_fish,
                np.full(shape=len(old_zero) + len(new_zero), fill_value=8),
            )
        total += len(old_fish) + len(new_fish)
    return total

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
puzzle.answer_b = answer
