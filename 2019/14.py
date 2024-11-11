import re
from collections import defaultdict
from functools import cache
from math import ceil

from aocd.models import Puzzle

YEAR = 2019
DAY = 14

puzzle = Puzzle(year=YEAR, day=DAY)


def calc_tot_needed(reactions, needed, used, stock):
    for q, n in needed.items():
        multiplier = ceil(n / reactions[q][0])
        multiplier = max(1, multiplier)
        stock[q] += reactions[q][0] * multiplier - n
        used[q] += n
        new_needed = {k: v * multiplier for k, v in reactions[q][1].items()}
        used, stock = calc_tot_needed(reactions, new_needed, used.copy(), stock.copy())
    return used, stock


def spend_min_ores(reactions, sstock):
    best_score = [-1]

    @cache
    def tmp(**stock):
        possible_to_buy = []
        for q, (n, needed) in reactions.items():
            for qq, nn in needed.items():
                if qq not in stock or stock[qq] < nn:
                    break
            else:
                possible_to_buy.append(q)
        for q in possible_to_buy:
            n, needed = reactions[q]
            new_stock = defaultdict(int)
            new_stock |= stock
            new_stock[q] += n
            for qq, nn in needed.items():
                new_stock[qq] -= nn
            if "FUEL" in stock:
                if stock["ORE"] > best_score[0]:
                    best_score[0] = stock["ORE"]
                yield stock["ORE"]
            if stock["ORE"] < best_score[0]:
                yield -1
            yield from tmp(**new_stock)
        yield -1

    return max(tmp(**sstock))


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
    stock = {k: v for k, v in stock.items() if v > 0}
    # used["ORE"] is an upper limit of how much ORE we might need
    new_stock = defaultdict(int)
    new_stock["ORE"] = used["ORE"]
    del reactions["ORE"]
    return used["ORE"] - spend_min_ores(reactions, new_stock)


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
