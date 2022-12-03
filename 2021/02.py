import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 2

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    data = [(line.split(" ")[0], int(line.split(" ")[1])) for line in data.split("\n")]
    horizontal = 0
    depth = 0
    for command, steps in data:
        if command == "forward":
            horizontal += steps
        elif command == "up":
            depth -= steps
        elif command == "down":
            depth += steps
        else:
            raise
    return horizontal * depth

assert a(puzzle.example_data) == 150
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    data = [(line.split(" ")[0], int(line.split(" ")[1])) for line in data.split("\n")]
    horizontal = 0
    depth = 0
    aim = 0
    for command, steps in data:
        if command == "forward":
            horizontal += steps
            depth += steps * aim
        elif command == "up":
            aim -= steps
        elif command == "down":
            aim += steps
        else:
            raise
    return horizontal * depth

assert b(puzzle.example_data) == 900
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
