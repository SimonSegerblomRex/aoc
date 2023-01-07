import functools
import operator
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

    dd = {tuple(k): list(pairwise((k[0], v, k[1]))) for k, v in insertion_rules.items()}
    tt = Counter(pairwise(template))
    for _ in range(steps):
        ndd = Counter()
        for k, v in tt.items():
            ndd[dd[k][0]] += v
            ndd[dd[k][1]] += v
        tt = ndd
    #[dd[k] k, v in for tt.items()]
    nn = Counter()
    for k, v in tt.items():
        nn[k[0]] += v
        nn[k[1]] += v
    counter = Counter(nn)
    ordered = counter.most_common()
    #breakpoint()
    hh = ordered[0][1] - ordered[-1][1]
    #breakpoint()
    # off by one for part b...
    return int(np.ceil(hh/2))

    pair_counts = {}
    element_counts = dict()
    for combo in insertion_rules:
        components = combo
        for _ in range(steps):
            new_components = []
            for pair in pairwise(components):
                new_components.append(pair[0])
                new_components.append(insertion_rules["".join(pair)])
            new_components.append(pair[1])
            breakpoint()
            components = new_components

            counter = Counter(components[:-1])
            element_counts[combo] = counter

        counter = Counter(components[:-1])
        element_counts[combo] = counter
    counter = functools.reduce(operator.add, (element_counts["".join(pair)] for pair in pairwise(template)))
    counter[template[-1]] += 1
    ordered = counter.most_common()
    return ordered[0][1] - ordered[-1][1]


example_answer = a(puzzle.example_data, steps=10)
print(example_answer)
assert example_answer == 1588
answer = a(puzzle.input_data, steps=10)
print("a:", answer)
assert answer == 2360


# Part b
example_answer = a(puzzle.example_data, steps=40)
print(example_answer)
assert example_answer == 2188189693529
answer = a(puzzle.input_data, steps=40)
print("b:", answer)
puzzle.answer_b = answer
