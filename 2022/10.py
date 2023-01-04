import re

import numpy as np
from aocd.models import Puzzle

np.set_printoptions(linewidth=180)

YEAR = 2022
DAY = 10

puzzle = Puzzle(year=YEAR, day=DAY)

EXAMPLE1 = """noop
addx 3
addx -5
"""

EXAMPLE2 = """addx 15
addx -11
addx 6
addx -3
addx 5
addx -1
addx -8
addx 13
addx 4
noop
addx -1
addx 5
addx -1
addx 5
addx -1
addx 5
addx -1
addx 5
addx -1
addx -35
addx 1
addx 24
addx -19
addx 1
addx 16
addx -11
noop
noop
addx 21
addx -15
noop
noop
addx -3
addx 9
addx 1
addx -3
addx 8
addx 1
addx 5
noop
noop
noop
noop
noop
addx -36
noop
addx 1
addx 7
noop
noop
noop
addx 2
addx 6
noop
noop
noop
noop
noop
addx 1
noop
noop
addx 7
addx 1
noop
addx -13
addx 13
addx 7
noop
addx 1
addx -33
noop
noop
noop
addx 2
noop
noop
noop
addx 8
noop
addx -1
addx 2
addx 1
noop
addx 17
addx -9
addx 1
addx 1
addx -3
addx 11
noop
noop
addx 1
noop
addx 1
noop
noop
addx -13
addx -19
addx 1
addx 3
addx 26
addx -30
addx 12
addx -1
addx 3
addx 1
noop
noop
noop
addx -9
addx 18
addx 1
addx 2
noop
noop
addx 9
noop
noop
noop
addx -1
addx 2
addx -37
addx 1
addx 3
noop
addx 15
addx -21
addx 22
addx -6
addx 1
noop
addx 2
addx 1
noop
addx -10
noop
noop
addx 20
addx 1
addx 2
addx 2
addx -6
addx -11
noop
noop
noop
"""

INSTRUCTIONS = {
    "addx": 2,
    "noop": 1,
}

# Part a
def a(data):
    X = 1
    x_incr = []
    total = 0
    instructions = data.splitlines()
    for c in range(1, 220 + 1):
        X += 0 if not x_incr else x_incr.pop(0)
        if instructions:
            instruction = instructions.pop(0)
            if instruction.startswith("noop"):
                x_incr.append(0)
                pass
            else:
                instruction, arg = instruction.split(" ")
                if instruction == "addx":
                    x_incr.append(0)
                    x_incr.append(int(arg))
        if c in [20, 60, 100, 140, 180, 220]:
            total += c * X
    return total


# a(EXAMPLE1)
example_answer = a(EXAMPLE2)
print(example_answer)
assert example_answer == 13140
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 16880

# Part b
def b(data):
    X = 1
    x_incr = []
    total = 0
    instructions = data.splitlines()
    crt = np.full((6, 40), 0, dtype=int)
    for c in range(1, 40 * 6 + 1):
        X += 0 if not x_incr else x_incr.pop(0)
        if instructions:
            instruction = instructions.pop(0)
            if instruction.startswith("noop"):
                x_incr.append(0)
                pass
            else:
                instruction, arg = instruction.split(" ")
                if instruction == "addx":
                    x_incr.append(0)
                    x_incr.append(int(arg))
        if c in [20, 60, 100, 140, 180, 220]:
            total += c * X
        sprite = [X - 1, X, X + 1]
        if (c - 1) % 40 in sprite:
            crt.ravel()[c - 1] = 1
    return crt


def print_image(x):
    print("")
    for i in range(x.shape[0]):
        print("".join("\u2588" if v else " " for v in x[i, :]))
    print("")


example_answer = b(EXAMPLE2)
print_image(example_answer)
answer = b(puzzle.input_data)
print_image(answer)
