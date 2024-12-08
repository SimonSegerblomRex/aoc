import datetime
import re
from collections import defaultdict

import numpy as np
from aocd.models import Puzzle
from itertools import permutations

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


def get_antinodes(c0, c1):
    d = c1 - c0
    return c0 - d, c1 + d


# Part a
def a(data):
    antennas = defaultdict(list)
    for i, line in enumerate(data.splitlines()):
        width = len(line)
        for j, c in enumerate(line):
            if c != ".":
                antennas[c].append(j + i * 1j)
    height = i + 1
    antinodes = set()
    for _, coords in antennas.items():
        for c0, c1 in permutations(coords, 2):
            aa = get_antinodes(c0, c1)
            tmp = set()
            for a in aa:
                if (0 <= a.real < width) and (0 <= a.imag < height):
                    tmp.add(a)
            antinodes |= tmp
    return len(antinodes)


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


def get_antinodes_b(c0, c1):
    d = c1 - c0
    out = set()
    for i in range(1000):
        out.add(c0 - i * d)
        out.add(c1 + i * d)
    return out


# Part b
def b(data):
    antennas = defaultdict(list)
    for i, line in enumerate(data.splitlines()):
        width = len(line)
        for j, c in enumerate(line):
            if c != ".":
                antennas[c].append(j + i * 1j)
    height = i + 1
    antinodes = set()
    for _, coords in antennas.items():
        for c0, c1 in permutations(coords, 2):
            aa = get_antinodes_b(c0, c1)
            tmp = set()
            for a in aa:
                if (0 <= a.real < width) and (0 <= a.imag < height):
                    tmp.add(a)
            antinodes |= tmp
    return len(antinodes)


answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
