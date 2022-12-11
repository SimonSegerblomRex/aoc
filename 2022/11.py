import re
from dataclasses import dataclass

import numpy as np
from aocd.models import Puzzle

YEAR = 2022
DAY = 11

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
def a(data, rounds, divide_by_three):
    data = re.finditer(PATTERN, data)
    monkeys = {
        int(m.group("nbr")): Monkey(
            nbr=int(m.group("nbr")),
            items=np.fromstring(m.group("starting_items"), dtype=int, sep=", ").tolist(),
            operation=m.group("operation"),
            divisible_by=int(m.group("divisble_by")),
            if_true_nbr=int(m.group("if_true_nbr")),
            if_false_nbr=int(m.group("if_false_nbr")),
        )
        for m in data
    }
    inspected_items = np.zeros(len(monkeys), dtype=int)
    magic = np.prod(list(set(m.divisible_by for m in monkeys.values())))
    for _ in range(rounds):
        for i in range(len(monkeys)):
            for old in monkeys[i].items:
                new = eval(monkeys[i].operation)
                if divide_by_three:
                    new //= 3
                new = new % magic
                if new % monkeys[i].divisible_by:
                    monkeys[monkeys[i].if_false_nbr].items.append(new)
                else:
                    monkeys[monkeys[i].if_true_nbr].items.append(new)
                inspected_items[i] += 1
            monkeys[i].items = []
    return np.prod(sorted(inspected_items)[-2:])


example_answer = a(puzzle.example_data, 20, divide_by_three=True)
print(example_answer)
assert example_answer == 10605
answer = a(puzzle.input_data, 20, divide_by_three=True)
print("a:", answer)
assert answer == 182293


# Part b
example_answer = a(puzzle.example_data, rounds=10000, divide_by_three=False)
print(example_answer)
assert example_answer == 2713310158
answer = a(puzzle.input_data, rounds=10000, divide_by_three=False)
print("b:", answer)
assert answer == 54832778815
