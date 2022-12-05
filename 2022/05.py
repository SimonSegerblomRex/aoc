import re

import numpy as np
from aocd.models import Puzzle

YEAR = 2022
DAY = 5

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    ugly_boxes = re.findall(".{4}", data[:data.find("1")], re.DOTALL)
    nbr_crates = int(re.findall("\d+\s+(\d+) \n", data)[0])
    boxes = {}
    for i in range(nbr_crates):
        boxes[i + 1] = [e.strip(" \n[]") for e in ugly_boxes[i::nbr_crates] if e.strip(" \n[]")]
    instructions = re.findall("move (\d+) from (\d+) to (\d+)", data)
    for instruction in instructions:
        nbr, fr, to = [int(e) for e in instruction]
        for _ in range(nbr):
            box = boxes[fr].pop(0)
            boxes[to].insert(0, box)
    return "".join([stack[0] for stack in boxes.values()])

example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == "CMZ"
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == "CNSZFDVLJ"


# Part b
def b(data):
    ugly_boxes = re.findall(".{4}", data[:data.find("1")], re.DOTALL)
    nbr_crates = int(re.findall("\d+\s+(\d+) \n", data)[0])
    boxes = {}
    for i in range(nbr_crates):
        boxes[i + 1] = [e.strip(" \n[]") for e in ugly_boxes[i::nbr_crates] if e.strip(" \n[]")]
    instructions = re.findall("move (\d+) from (\d+) to (\d+)", data)
    for instruction in instructions:
        nbr, fr, to = [int(e) for e in instruction]
        for i in range(nbr):
            box = boxes[fr].pop(0)
            boxes[to].insert(i, box)
    return "".join([stack[0] for stack in boxes.values()])

example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == "MCD"
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == "QNDWLMGNS"
