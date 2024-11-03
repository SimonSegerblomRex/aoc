from itertools import permutations

import numpy as np
from aocd.models import Puzzle

YEAR = 2019
DAY = 7

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def run(codes, inputs, pos=0):
    while True:
        instruction = codes[pos]
        opcode = instruction % 100
        finished = False
        def get_mode(instruction, i):
            return instruction // 10**(i + 2) & 0b11
        def get_idx(instruction, i):
            mode = get_mode(instruction, i)
            if mode == 0:
                return codes[pos + i + 1]
            if mode == 1:
                return pos + i + 1
            if mode == 2:
                return codes[pos + i + 1]  # hmm
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
    phases_permutations = permutations(range(5), 5)
    scores = []
    for phases in phases_permutations:
        input0 = 0
        for phase in phases:
            input0, *_ = run(codes.copy(), [phase, input0])
        scores.append(input0)
    return max(scores)


answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 24405


# Part b
def b(data):
    codes = list(map(int, data.split(",")))
    phases_permutations = permutations(range(5, 10), 5)
    scores = []
    for phases in phases_permutations:
        codes_copies = [codes.copy() for _ in range(5)]
        pos_cache = [0] * 5
        input0 = 0
        finished = False
        for i, phase in enumerate(phases):
            input0, pos, finished = run(codes_copies[i], [phase, input0])
            pos_cache[i] = pos
        while not finished:
            for i in range(5):
                input0, pos, finished = run(codes_copies[i], [input0], pos=pos_cache[i])
                pos_cache[i] = pos
        scores.append(input0)
    return max(scores)


answer = b(puzzle.input_data)
assert answer < 9065837
print("b:", answer)
assert answer == 8271623
