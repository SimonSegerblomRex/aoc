from collections import defaultdict
from itertools import permutations

import numpy as np
from aocd.models import Puzzle

YEAR = 2019
DAY = 11

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def run(codes, inputs, pos=0):
    relative_base = 0
    while True:
        instruction = codes[pos]
        opcode = instruction % 100
        finished = False
        def get_mode(instruction, i):
            return int(("0000" + str(instruction))[-3 - i])
        def get_idx(instruction, i):
            mode = get_mode(instruction, i)
            if mode == 0:
                return codes[pos + i + 1]
            if mode == 1:
                return pos + i + 1
            if mode == 2:
                return relative_base + codes[pos + i + 1]
            raise NotImplementedError(f"Unsupported {mode=}")
        if opcode == 1:
            idx0 = get_idx(instruction, 0)
            idx1 = get_idx(instruction, 1)
            idx2 = get_idx(instruction, 2)
            codes[idx2] = codes[idx0] + codes[idx1]
            pos += 4
        elif opcode == 2:
            idx0 = get_idx(instruction, 0)
            idx1 = get_idx(instruction, 1)
            idx2 = get_idx(instruction, 2)
            codes[idx2] = codes[idx0] * codes[idx1]
            pos += 4
        elif opcode == 3:
            idx0 = get_idx(instruction, 0)
            codes[idx0] = inputs.pop(0)
            pos += 2
        elif opcode == 4:
            idx0 = get_idx(instruction, 0)
            inputs.append(codes[idx0])
            pos += 2
            break
        elif opcode == 5:
            idx0 = get_idx(instruction, 0)
            idx1 = get_idx(instruction, 1)
            if codes[idx0] != 0:
                pos = codes[idx1]
            else:
                pos += 3
        elif opcode == 6:
            idx0 = get_idx(instruction, 0)
            idx1 = get_idx(instruction, 1)
            if codes[idx0] == 0:
                pos = codes[idx1]
            else:
                pos += 3
        elif opcode == 7:
            idx0 = get_idx(instruction, 0)
            idx1 = get_idx(instruction, 1)
            idx2 = get_idx(instruction, 2)
            if codes[idx0] < codes[idx1]:
                codes[idx2] = 1
            else:
                codes[idx2] = 0
            pos += 4
        elif opcode == 8:
            idx0 = get_idx(instruction, 0)
            idx1 = get_idx(instruction, 1)
            idx2 = get_idx(instruction, 2)
            if codes[idx0] == codes[idx1]:
                codes[idx2] = 1
            else:
                codes[idx2] = 0
            pos += 4
        elif opcode == 9:
            idx0 = get_idx(instruction, 0)
            relative_base += codes[idx0]
            pos += 2
        elif opcode == 99:
            pos += 1
            finished = True
            break
        else:
            print(f"Unknown {opcode=}")
            breakpoint()
    return inputs.pop(), pos, finished


def a(data):
    codes = list(map(int, data.split(",")))
    codes.extend([0]*1000)
    coords = defaultdict(int)  # default to 0, black
    pos = 0 + 0j
    dir = 0 + 1j
    cpos = 0
    finished = False
    while not finished:
        inp = coords[pos]
        col, cpos, finished = run(codes, [inp], cpos)
        if finished:
            break
        right, cpos, finished = run(codes, [inp], cpos)
        dir = dir * (-1j if right else 1j)
        coords[pos] = col
        pos += dir
    return len(coords)


answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    breakpoint()


answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer