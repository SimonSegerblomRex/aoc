import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    rows = np.array([[ord(c) for c in line] for line in data.splitlines()], dtype=int)
    s = 0
    for x in rows, np.rot90(rows):
        for line in x:
            line = "".join(chr(n) for n in line)
            s += line.count("XMAS")
            s += line.count("SAMX")
        for i in range(-rows.shape[0] + 1, rows.shape[0]):
            line = np.diag(x, i)
            line = "".join(chr(n) for n in line)
            s += line.count("XMAS")
            s += line.count("SAMX")
    return s


answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    rows = np.array([[ord(c) for c in line] for line in data.splitlines()], dtype=int)
    M = 100
    A = 100000
    S = 10000000
    kernels = [
        [
            [M, 0, S],
            [0, A, 0],
            [M, 0, S],
        ],
        [
            [M, 0, M],
            [0, A, 0],
            [S, 0, S],
        ],
        [
            [S, 0, S],
            [0, A, 0],
            [M, 0, M],
        ],
        [
            [S, 0, M],
            [0, A, 0],
            [S, 0, M],
        ],
    ]
    key = 2*ord("M")*M + ord("A")*A + 2*ord("S")*S
    from scipy.signal import convolve2d
    s = 0
    for kernel in kernels:
        tmp = convolve2d(rows, kernel)
        s += (tmp == key).sum()
    return s


answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
