import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, boxes):
    instructions = re.findall("move (\d+) from (\d+) to (\d+)", data)
    for instruction in instructions:
        nbr, fr, to = [int(e) for e in instruction]
        for _ in range(nbr):
            box = boxes[fr].pop(0)
            boxes[to].insert(0, box)
    return "".join([stack[0] for stack in boxes.values()])

boxes_example = {
    1: ["N", "Z"],
    2: ["D", "C", "M"],
    3: ["P"]
}
boxes = {
    1: ["N", "V", "C", "S"],
    2: ["S", "N", "H", "J", "M", "Z"],
    3: ["D", "N", "J", "G", "T", "C", "M"],
    4: ["M", "R", "W", "J", "F", "D", "T"],
    5: ["H", "F", "P"],
    6: ["J", "H", "Z", "T", "C"],
    7: ["Z", "L", "S", "F", "Q", "R", "P", "D"],
    8: ["W", "P", "F", "D", "H", "L", "S", "C"],
    9: ["Z", "G", "N", "F", "P", "M", "S", "D"],
}
example_answer = a(puzzle.example_data, boxes_example)
print(example_answer)
assert example_answer == "CMZ"
answer = a(puzzle.input_data, boxes)
print("a:", answer)
#puzzle.answer_a = answer


boxes_example = {
    1: ["N", "Z"],
    2: ["D", "C", "M"],
    3: ["P"]
}
boxes = {
    1: ["N", "V", "C", "S"],
    2: ["S", "N", "H", "J", "M", "Z"],
    3: ["D", "N", "J", "G", "T", "C", "M"],
    4: ["M", "R", "W", "J", "F", "D", "T"],
    5: ["H", "F", "P"],
    6: ["J", "H", "Z", "T", "C"],
    7: ["Z", "L", "S", "F", "Q", "R", "P", "D"],
    8: ["W", "P", "F", "D", "H", "L", "S", "C"],
    9: ["Z", "G", "N", "F", "P", "M", "S", "D"],
}
# Part b
def b(data, boxes):
    instructions = re.findall("move (\d+) from (\d+) to (\d+)", data)
    for instruction in instructions:
        nbr, fr, to = [int(e) for e in instruction]
        for i in range(nbr):
            box = boxes[fr].pop(0)
            boxes[to].insert(i, box)
    return "".join([stack[0] for stack in boxes.values()])

example_answer = b(puzzle.example_data, boxes_example)
print(example_answer)
assert example_answer == "MCD"
answer = b(puzzle.input_data, boxes)
print("b:", answer)
puzzle.answer_b = answer
