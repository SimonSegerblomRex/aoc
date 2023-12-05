import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    lines = iter(data.splitlines())
    seeds = np.fromstring(next(lines).split(":")[1], dtype=int, sep=" ")
    big_map = []
    for match in re.finditer(r"(?P<from>\w+)-to-(?P<to>\w+) map:\n(?P<maps>[\d\s\n]+)", data):
        maps = np.fromstring(match["maps"], dtype=int, sep=" ")
        maps.shape = (-1, 3)
        big_map.append(maps)
    ll = []
    for s in seeds:
        t = s
        for m in big_map:
            for r in m:
                if r[1] <= t < r[1] + r[2]:
                    t = r[0] + t - r[1]
                    break
        ll.append(t)
    return min(ll)
    breakpoint()


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    #print(data)
    print("BBB")
    breakpoint()


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
