import datetime
import re
import itertools

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    total = 0
    for line in data.splitlines():
        c = 0
        springs, group_sizes = line.split(" ")
        group_sizes = np.fromstring(group_sizes, sep=",", dtype=int)
        nbr_of_springs = sum(group_sizes)
        unknowns = springs.count("?")
        nbr_of_missing_springs = nbr_of_springs - springs.count("#")
        for combo in itertools.permutations("#" * nbr_of_missing_springs + "." * (unknowns - nbr_of_missing_springs), unknowns):
            tmp = np.array(list(springs))
            tmp[tmp == "?"] = combo
            tmp = "".join(tmp)
            groups = [g for g in tmp.split(".") if g]
            sizes = [len(g) for g in groups]
            if (len(group_sizes) == len(sizes)) and (group_sizes == sizes).all():
                #print(combo, tmp)
                c += 1
        total += c
        breakpoint()


example = """???.### 1,1,3
.??..??...?##. 1,1,3
?###???????? 3,2,1"""
example_answer = a(example)


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
