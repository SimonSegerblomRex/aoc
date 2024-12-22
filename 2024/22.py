import datetime
import re
from itertools import product

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    s = 0
    for n in data.split():
        for _ in range(2000):
            n = int(n)
            n ^= 64 * n
            n %= 16777216
            n ^= n // 32
            n %= 16777216
            n ^= 2048 * n
            n %= 16777216
        s += n
    return s



example = """1
10
100
2024"""
answer = a(example)
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 12664695565


# Part b
def b(data):
    #numbers = []
    prices = []
    diff = []
    for n in data.split():
        n = int(n)
        tmp_n = [n]
        tmp_p = [n % 10]
        tmp_d = [0]
        for _ in range(2000):
            n = int(n)
            n ^= 64 * n
            n %= 16777216
            n ^= n // 32
            n %= 16777216
            n ^= 2048 * n
            n %= 16777216
            tmp_d.append(n % 10 - tmp_p[-1])
            tmp_p.append(n % 10)
            tmp_n.append(n)
        #numbers.append(tmp_n)
        prices.append(tmp_p)
        diff.append(tmp_d)
    seqs = set()
    for p, d in zip(prices, diff):
        for i in range(1, len(p) - 3):
            seqs.add(tuple(d[i:i + 4]))
    bananas = {}
    #for seq in product(range(-9, 9 + 1), repeat=4):
    for seq in seqs:
        s = 0
        for p, d in zip(prices, diff):
            for i in range(1, len(p) - 3):
                if tuple(d[i:i + 4]) == seq:
                    s += p[i + 3]
                    break
        bananas[seq] = s
    return max(bananas.values())


example = """1
2
3
2024"""
answer = b(example)
print(answer)
assert answer == 23
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
