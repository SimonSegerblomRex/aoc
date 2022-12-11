import datetime
import inspect
import re
from dataclasses import dataclass

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)

PATTERN = r"""Monkey (?P<nbr>\d):
  Starting items: (?P<starting_items>.*)
  Operation: new = (?P<operation>.+)
  Test: divisible by (?P<divisble_by>\d+)
    If true: throw to monkey (?P<if_true_nbr>\d+)
    If false: throw to monkey (?P<if_false_nbr>\d+)"""

@dataclass
class Monkey:
    nbr: int
    items: list
    operation: str
    divisible_by: int
    if_true_nbr: int
    if_false_nbr: int

# Part a
def a(data, rounds):
    data = re.findall(PATTERN, data)
    monkeys = {int(e[0]): Monkey(nbr=int(e[0]), items=np.fromstring(e[1], dtype=int, sep=", ").tolist(), operation=e[2], divisible_by=int(e[3]), if_true_nbr=int(e[4]), if_false_nbr=int(e[5])) for e in data}
    inspected_items = [0,] * len(monkeys)
    for _ in range(rounds):
        for i in range(len(monkeys)):
            for item in monkeys[i].items:
                old = item
                new = eval(monkeys[i].operation)
                new //= 3
                if new % monkeys[i].divisible_by:
                    monkeys[monkeys[i].if_false_nbr].items.append(new)
                else:
                    monkeys[monkeys[i].if_true_nbr].items.append(new)
                inspected_items[i] += 1
            monkeys[i].items = []
    return np.prod(sorted(inspected_items)[-2:])

example_answer = a(puzzle.example_data, 20)
print(example_answer)
assert example_answer == 10605
answer = a(puzzle.input_data, 20)
print("a:", answer)
assert answer == 182293


# Part b
def b(data, rounds):
    data = re.findall(PATTERN, data)
    monkeys = {int(e[0]): Monkey(nbr=int(e[0]), items=np.fromstring(e[1], dtype=int, sep=", ").tolist(), operation=e[2], divisible_by=int(e[3]), if_true_nbr=int(e[4]), if_false_nbr=int(e[5])) for e in data}
    inspected_items = [0,] * len(monkeys)
    magic = np.prod(list(set(m.divisible_by for m in monkeys.values())))
    for r in range(rounds):
        for i in range(len(monkeys)):
            for item in monkeys[i].items:
                old = item
                new = eval(monkeys[i].operation)
                new = new % magic
                if new % monkeys[i].divisible_by:
                    monkeys[monkeys[i].if_false_nbr].items.append(new)
                else:
                    monkeys[monkeys[i].if_true_nbr].items.append(new)
                inspected_items[i] += 1
            monkeys[i].items = []
    return np.prod(sorted(inspected_items)[-2:])

example_answer = b(puzzle.example_data, rounds=10000)
print(example_answer)
assert example_answer == 2713310158
answer = b(puzzle.input_data, rounds=10000)
print("b:", answer)
assert answer == 54832778815
