import re
from collections import defaultdict
from functools import cache
from math import ceil

from aocd.models import Puzzle

YEAR = 2019
DAY = 14

puzzle = Puzzle(year=YEAR, day=DAY)


def calc_needed(reactions):
    reactions["ORE"] = (1, {"ORE": 1})
    needed = defaultdict(int, reactions["FUEL"][1])
    stock = defaultdict(int)
    stock["ORE"] = 1e90
    used = defaultdict(int)
    while needed:
        q = list(needed)[0]
        n = needed.pop(q)
        used[q] += n
        wanted = n
        if wanted <= stock[q]:
            stock[q] -= wanted
        else:
            wanted -= stock[q]
            chunk = reactions[q][0]
            multiplier = ceil(wanted / chunk)
            multiplier = max(1, multiplier)
            stock[q] += chunk * multiplier - n
            for k, v in reactions[q][1].items():
                needed[k] += max(v * multiplier, 0)
            needed = defaultdict(int, {k: v for k, v in needed.items() if v >0})
    return used["ORE"]


# Part a
def a(data):
    reactions = {}
    for line in data.splitlines():
        in_ = re.findall("(\d+) ([A-Z]+)", line)
        out = in_.pop(-1)
        reactions[out[1]] = int(out[0]), {q: int(n) for n, q in in_}
    reactions["ORE"] = (1, {})
    return calc_needed(reactions)


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
    breakpoint()


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
