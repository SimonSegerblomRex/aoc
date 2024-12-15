import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    grid, moves = data.split("\n\n")
    wall = set()
    boxes = set()
    pos = None
    for i, line in enumerate(grid.splitlines()):
        width = len(line)
        for j, c in enumerate(line):
            if c == "#":
                wall.add(j + i*1j)
            elif c == "O":
                boxes.add(j + i*1j)
            elif c == "@":
                pos = j + i*1j
    height = i + 1
    move2dir = {
        ">": 1,
        "^": -1j,
        "<": -1,
        "v": 1j,
    }
    for move in moves.replace("\n", ""):
        dir = move2dir[move]
        if pos + dir in wall:
            continue
        if pos + dir not in boxes:
            pos += dir
            continue
        boxes_to_move = 0
        pos_to_check = pos + dir
        while True:
            if pos_to_check in boxes:
                boxes_to_move += 1
                pos_to_check += dir
            elif pos_to_check in wall:
                boxes_to_move = 0
                break
            else:
                break
        if boxes_to_move > 0:
            pos += dir
            box_to_move = pos
            boxes.remove(pos)
            boxes.add(pos + boxes_to_move * dir)
    s = 0
    for box in boxes:
        s += box.real + 100 * box.imag
    return s
    breakpoint()


answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    print(data)
    breakpoint()


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
