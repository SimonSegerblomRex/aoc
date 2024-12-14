import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, width=101, height=103, seconds=100):
    robot_pos = []
    robot_dir = []
    for line in data.splitlines():
        x = [int(n) for n in re.findall(r"(-?\d+)", line)]
        pos = x[0] + x[1]*1j
        dir = x[2] + x[3]*1j
        robot_pos.append(pos)
        robot_dir.append(dir)
    for _ in range(seconds):
        for i, (pos, dir) in enumerate(zip(robot_pos, robot_dir)):
            new_pos = pos + dir
            new_pos = (new_pos.real % width) + (new_pos.imag % height) * 1j
            robot_pos[i] = new_pos
    scores = [0, 0, 0 ,0]
    for pos in robot_pos:
        if (pos.real < ((width - 1) // 2)) and (pos.imag < ((height - 1) // 2)):
            scores[0] += 1
        elif (pos.real > ((width - 1) // 2)) and (pos.imag < ((height - 1) // 2)):
            scores[1] += 1
        elif (pos.real < ((width - 1) // 2)) and pos.imag > (((height - 1) // 2)):
            scores[2] += 1
        elif (pos.real > ((width - 1) // 2)) and pos.imag > (((height - 1) // 2)):
            scores[3] += 1
    print(scores)
    return scores[0] * scores[1] * scores[2] * scores[3]


example = """p=0,4 v=3,-3
p=6,3 v=-1,-3
p=10,3 v=-1,2
p=2,0 v=2,-1
p=0,0 v=1,3
p=3,0 v=-2,-2
p=7,6 v=-1,-3
p=3,0 v=-1,-2
p=9,3 v=2,3
p=7,3 v=-1,2
p=2,4 v=2,-3
p=9,5 v=-3,-3"""
answer = a(example, width=11, height=7)
print(answer)
assert answer == 12
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer
breakpoint()

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
