import re
from collections import defaultdict
from functools import cache
from math import ceil

from aocd.models import Puzzle

YEAR = 2019
DAY = 14

puzzle = Puzzle(year=YEAR, day=DAY)


def calc_needed(reactions, stock, needed):
    used = defaultdict(int)
    flag = 0
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
            needed = defaultdict(int, {k: v for k, v in needed.items() if v > 0})
            if needed["ORE"] > stock["ORE"]:
                flag = 1
                break
    return used, stock, flag


def get_reactions(data):
    reactions = {}
    for line in data.splitlines():
        in_ = re.findall("(\d+) ([A-Z]+)", line)
        out = in_.pop(-1)
        reactions[out[1]] = int(out[0]), {q: int(n) for n, q in in_}
    reactions["ORE"] = (1, {"ORE": 1})
    return reactions



# Part a
def a(data):
    reactions = get_reactions(data)
    stock = defaultdict(int)
    stock["ORE"] = int(1e12)
    needed = defaultdict(int, {"FUEL": 1})
    return calc_needed(reactions, stock, needed)[0]["ORE"]


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 202617


# Part b
def b(data):
    reactions = get_reactions(data)
    stock = defaultdict(int)
    stock["ORE"] = int(1e12)
    fuel_chunk = int(1e11)
    fuel = 0
    while True:
        needed = defaultdict(int, {"FUEL": fuel_chunk})
        stock["FUEL"] = 0
        backup_stock = stock.copy()
        used, stock, flag = calc_needed(reactions, stock, needed)
        if flag > 0:
            stock = backup_stock
            if fuel_chunk > 1:
                fuel_chunk //= 10
            else:
                break
        else:
            fuel += fuel_chunk
    return fuel


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b


answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 7863863
