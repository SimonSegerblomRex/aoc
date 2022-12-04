import datetime
import re

from aocd.models import Puzzle

YEAR = 2022
DAY = 4

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    data = re.findall("(\d+)-(\d+),(\d+)-(\d+)", data)
    data = [
        set(range(int(r[0]), int(r[1]) + 1)).issubset(
            set(range(int(r[2]), int(r[3]) + 1))
        )
        or set(range(int(r[2]), int(r[3]) + 1)).issubset(
            set(range(int(r[0]), int(r[1]) + 1))
        )
        for r in data
    ]
    return sum(data)


assert a(puzzle.example_data) == 2
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 571


# Part b
def b(data):
    data = re.findall("(\d+)-(\d+),(\d+)-(\d+)", data)
    data = [
        set(range(int(r[0]), int(r[1]) + 1)).intersection(
            set(range(int(r[2]), int(r[3]) + 1))
        )
        for r in data
    ]
    return sum([d != set() for d in data])


assert b(puzzle.example_data) == 4
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 917
