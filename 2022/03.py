import numpy as np
from aocd.models import Puzzle

YEAR = 2022
DAY = 3

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    rucksacks = data.split("\n")
    s = 0
    for rucksack in rucksacks:
        size = len(rucksack)
        c1, c2 = rucksack[: size // 2], rucksack[size // 2 :]
        for c in c1:
            if c in c2:
                break
        if c.islower():
            p = ord(c) - ord("a") + 1
        else:
            p = ord(c) - 38
        s += p
    return s


assert a(puzzle.example_data) == 157
answer = a(puzzle.input_data)
print("a:", answer)


# Part b
def b(data):
    rucksacks = data.split("\n")
    groups = np.array(rucksacks)
    groups.shape = (-1, 3)
    s = 0
    for group in groups:
        for c in group[0]:
            if (c in group[1]) and (c in group[2]):
                break
        if c.islower():
            p = ord(c) - ord("a") + 1
        else:
            p = ord(c) - 38
        s += p
    return s


assert b(puzzle.example_data) == 70
answer = b(puzzle.input_data)
print("b:", answer)
