import datetime
import re
from collections import defaultdict

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    connections = defaultdict(list)
    for line in data.split():
        c0, c1 = line.split("-")
        connections[c0].append(c1)
        connections[c1].append(c0)
    s = 0
    hmm = set()
    for c0, conns0 in connections.items():
        for c1, conns1 in connections.items():
            if c0 == c1:
                continue
            if c1 not in conns0:
                continue
            if c0 not in conns1:
                continue
            for c2 in set(conns0) & set(conns1):
                tmp = set([c0, c1, c2])
                if len(tmp) == 3:
                    ok = False
                    for c in tmp:
                        if c[0] == "t":
                            ok = True
                    if ok:
                        hmm.add(tuple(sorted(list(tmp))))
    return len(hmm)


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {7})")
        assert example_answer == 7
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    breakpoint()


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
