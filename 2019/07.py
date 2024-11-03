from itertools import permutations

import numpy as np
from aocd.models import Puzzle

YEAR = 2019
DAY = 7

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def run(codes, inputs):
    pos = 0
    while True:
        instruction = codes[pos]
        opcode = instruction % 100
        if opcode == 1:
            idx0 = pos + 1 if (instruction // 100) % 2 else codes[pos + 1]
            idx1 = pos + 2 if (instruction // 1000) % 2 else codes[pos + 2]
            idx2 = pos + 3 if (instruction // 10000) % 2 else codes[pos + 3]
            codes[idx2] = codes[idx0] + codes[idx1]
            pos += 4
        elif opcode == 2:
            idx0 = pos + 1 if (instruction // 100) % 2 else codes[pos + 1]
            idx1 = pos + 2 if (instruction // 1000) % 2 else codes[pos + 2]
            idx2 = pos + 3 if (instruction // 10000) % 2 else codes[pos + 3]
            codes[idx2] = codes[idx0] * codes[idx1]
            pos += 4
        elif opcode == 3:
            idx0 =  pos + 1 if (instruction // 100) % 2 else codes[pos + 1]
            codes[idx0] = inputs.pop(0)
            pos += 2
        elif opcode == 4:
            idx0 =  pos + 1 if (instruction // 100) % 2 else codes[pos + 1]
            inputs.append(codes[idx0])
            pos += 2
        elif opcode == 5:
            idx0 = pos + 1 if (instruction // 100) % 2 else codes[pos + 1]
            idx1 = pos + 2 if (instruction // 1000) % 2 else codes[pos + 2]
            if codes[idx0] != 0:
                pos = codes[idx1]
            else:
                pos += 3
        elif opcode == 6:
            idx0 = pos + 1 if (instruction // 100) % 2 else codes[pos + 1]
            idx1 = pos + 2 if (instruction // 1000) % 2 else codes[pos + 2]
            if codes[idx0] == 0:
                pos = codes[idx1]
            else:
                pos += 3
        elif opcode == 7:
            idx0 = pos + 1 if (instruction // 100) % 2 else codes[pos + 1]
            idx1 = pos + 2 if (instruction // 1000) % 2 else codes[pos + 2]
            idx2 = pos + 3 if (instruction // 10000) % 2 else codes[pos + 3]
            if codes[idx0] < codes[idx1]:
                codes[idx2] = 1
            else:
                codes[idx2] = 0
            pos += 4
        elif opcode == 8:
            idx0 = pos + 1 if (instruction // 100) % 2 else codes[pos + 1]
            idx1 = pos + 2 if (instruction // 1000) % 2 else codes[pos + 2]
            idx2 = pos + 3 if (instruction // 10000) % 2 else codes[pos + 3]
            if codes[idx0] == codes[idx1]:
                codes[idx2] = 1
            else:
                codes[idx2] = 0
            pos += 4
        elif opcode == 99:
            break
        else:
            print(f"Unknown {opcode=}")
            breakpoint()
    return inputs.pop()


def a(data):
    codes = list(map(int, data.split(",")))
    phases_permutations = permutations(range(5), 5)
    scores = []
    for phases in phases_permutations:
        input0 = 0
        for phase in phases:
            input0 = run(codes.copy(), [phase, input0])
        scores.append(input0)
    return max(scores)


answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 24405

# Part b
def b(data):
    breakpoint()


answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
