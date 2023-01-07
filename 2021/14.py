import re
from collections import Counter
from more_itertools import pairwise

import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 14

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, steps):
    template, _, *insertion_rules = data.splitlines()
    insertion_rules = dict(rule.split(" -> ") for rule in insertion_rules)
    components = template
    for _ in range(steps):
        new_components = []
        for pair in pairwise(components):
            new_components.append(pair[0])
            new_components.append(insertion_rules["".join(pair)])
        new_components.append(pair[1])
        components = new_components
    counter = Counter(components)
    ordered = counter.most_common()
    return ordered[0][1] - ordered[-1][1]


example_answer = a(puzzle.example_data, steps=10)
print(example_answer)
assert example_answer == 1588
answer = a(puzzle.input_data, steps=10)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    exit()


example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == ...
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
