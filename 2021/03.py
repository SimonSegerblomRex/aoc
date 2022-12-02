import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 3

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(input):
    bits = len(input.split("\n")[0])
    print(bits)
    input = np.vstack([np.frombuffer(n.encode(), dtype=np.uint8) - ord("0") for n in input.split("\n")])
    input = input[:, -bits:]
    nbr_ones = input.sum(axis=0)
    gamma = (nbr_ones > input.shape[0] // 2).astype(np.uint8)
    epsilon = 1 - gamma
    gamma = int("".join(str(b) for b in gamma.tolist()), 2)
    epsilon = int("".join(str(b) for b in epsilon.tolist()), 2)
    return gamma * epsilon

assert a(puzzle.example_data) == 198
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(input):
    exit()

assert b(puzzle.example_data) == ...
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
