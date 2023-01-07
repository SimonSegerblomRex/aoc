from collections import Counter

from aocd.models import Puzzle
from more_itertools import pairwise

YEAR = 2021
DAY = 14

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, steps):
    template, _, *insertion_rules = data.splitlines()
    insertion_rules = dict(rule.split(" -> ") for rule in insertion_rules)

    pair_to_pairs = {
        tuple(k): list(pairwise((k[0], v, k[1]))) for k, v in insertion_rules.items()
    }
    pairs = Counter(pairwise(template))
    for _ in range(steps):
        new_pairs = Counter()
        for k, v in pairs.items():
            new_pairs[pair_to_pairs[k][0]] += v
            new_pairs[pair_to_pairs[k][1]] += v
        pairs = new_pairs

    elements = Counter()
    for k, v in pairs.items():
        elements[k[0]] += v
        elements[k[1]] += v
    counter = Counter(elements)

    counter[template[0]] += 1
    counter[template[-1]] += 1

    ordered = counter.most_common()
    return (ordered[0][1] - ordered[-1][1]) // 2


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
assert answer == 2967977072188
