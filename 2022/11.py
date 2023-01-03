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
    If true: throw to monkey (?P<true_nbr>\d+)
    If false: throw to monkey (?P<false_nbr>\d+)"""


@dataclass
class Monkey:
    nbr: int
    items: list
    operation: str
    divisible_by: int
    true_nbr: int
    false_nbr: int
    inspected: int = 0


# Part a
def a(data, rounds, divide_by_three):
    data = re.finditer(PATTERN, data)
    monkeys = [
        Monkey(
            nbr=int(m.group("nbr")),
            items=np.fromstring(
                m.group("starting_items"), dtype=int, sep=", "
            ).tolist(),
            operation=eval(f"lambda old: {m.group('operation')}"),
            divisible_by=int(m.group("divisble_by")),
            true_nbr=int(m.group("true_nbr")),
            false_nbr=int(m.group("false_nbr")),
        )
        for m in data
    ]
    magic = np.prod(list(set(m.divisible_by for m in monkeys)))
    for _ in range(rounds):
        for monkey in monkeys:
            for old in monkey.items:
                new = monkey.operation(old)
                if divide_by_three:
                    new //= 3
                new %= magic
                throw_to = (
                    monkey.false_nbr if (new % monkey.divisible_by) else monkey.true_nbr
                )
                monkeys[throw_to].items.append(new)
            monkey.inspected += len(monkey.items)
            monkey.items = []
    return np.prod(sorted(m.inspected for m in monkeys)[-2:])


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
