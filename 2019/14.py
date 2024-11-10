import re
from collections import defaultdict
from fractions import Fraction
from math import ceil

import numpy as np
from aocd.models import Puzzle

YEAR = 2019
DAY = 14

puzzle = Puzzle(year=YEAR, day=DAY)


def calc_frac_tot_needed(reactions, needed, used):
    for q, n in needed.items():
        f = Fraction(n, reactions[q][0])
        used[q] += n
        used = calc_frac_tot_needed(reactions, {k: v * f for k, v in reactions[q][1].items()}, used.copy())
    return used


def calc_tot_needed(reactions, needed, used, stock):
    for q, n in needed.items():
        multiplier = ceil(n / reactions[q][0])
        multiplier = max(1, multiplier)
        stock[q] += reactions[q][0] * multiplier - n
        used[q] += n
        #new_needed = {k: max(0, v * multiplier - stock[k]) for k, v in reactions[q][1].items()}
        #new_needed = {k: v for k, v in reactions[q][1].items() if v > 0}
        new_needed = {k: v * multiplier for k, v in reactions[q][1].items()}
        used, stock = calc_tot_needed(reactions, new_needed, used.copy(), stock.copy())
    return used, stock



# Part a
def a(data):
    reactions = {}
    for line in data.splitlines():
        in_ = re.findall("(\d+) ([A-Z]+)", line)
        out = in_.pop(-1)
        reactions[out[1]] = int(out[0]), {q: int(n) for n, q in in_}
    reactions["ORE"] = (1, {})
    needed = reactions["FUEL"][1]
    used = defaultdict(int)
    stock = defaultdict(int)
    used, stock = calc_tot_needed(reactions, needed, used, stock)
    # calc ore in stock
    extra = 0
    for k, v in stock.items():
        if len(reactions[k][1]) == 1 and "ORE" in reactions[k][1]:
            m = v // reactions[k][0]
            extra -= m*reactions[k][1]["ORE"]
    if 0:
        while True:
            missing = {k: ceil(used[k] / reactions[k][0]) * reactions[k][0] - used[k] for k, v in used.items()}
            missing = {k: v for k, v in missing.items() if v > 0}
            breakpoint()
            if not missing:
                break
            used = calc_tot_needed(reactions, missing, used)
    return used["ORE"] + extra


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        #assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
#puzzle.answer_a = answer


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
