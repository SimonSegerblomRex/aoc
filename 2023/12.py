from functools import cache

import numpy as np
from aocd.models import Puzzle
from more_itertools import distinct_permutations

YEAR = 2023
DAY = 12

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    total = 0
    for line in data.splitlines():
        c = 0
        springs, group_sizes = line.split(" ")
        if not springs:
            return 1
        group_sizes = np.fromstring(group_sizes, sep=",", dtype=int)
        nbr_damaged = sum(group_sizes)
        nbr_unknowns = springs.count("?")
        nbr_damaged_in_unknowns = nbr_damaged - springs.count("#")
        nbr_operational_in_unknowns = nbr_unknowns - nbr_damaged_in_unknowns
        for combo in distinct_permutations(
            "#" * nbr_damaged_in_unknowns + "." * nbr_operational_in_unknowns
        ):
            tmp = np.array(list(springs))
            tmp[tmp == "?"] = combo
            tmp = "".join(tmp)
            groups = [g for g in tmp.split(".") if g]
            sizes = [len(g) for g in groups]
            if (len(group_sizes) == len(sizes)) and (group_sizes == sizes).all():
                # print(combo, tmp)
                c += 1
        total += c
    return total


examples = [
    "???.### 1,1,3",
    ".??..??...?##. 1,1,3",
    "?#?#?#?#?#?#?#? 1,3,1,6",
    "????.#...#... 4,1,1",
    "????.######..#####. 1,6,5",
    "?###???????? 3,2,1",
]
example_answers_a = [1, 4, 1, 1, 4, 10]
for example, example_answer_a in zip(examples, example_answers_a):
    example_answer = a(example)
    print(f"Example answer: {example_answer} (expecting: {example_answer_a})")
    assert example_answer == example_answer_a
if 0:
    answer = a(puzzle.input_data)
    print("a:", answer)
    assert answer == 7653


# Part b
@cache
def combos(springs, sizes, curr_size, curr_target):
    springs = list(springs)
    sizes = list(sizes)
    while springs:
        c = springs.pop(0)
        if c == ".":
            if curr_size:
                if curr_size != curr_target:
                    return 0
                if not sizes:
                    if "#" not in springs:
                        return 1
                    return 0
                curr_target = sizes.pop(0)
                curr_size = 0
        elif c == "#":
            curr_size += 1
            if curr_size > curr_target:
                return 0
        elif c == "?":
            # case: .
            if curr_size in (0, curr_target):
                c1 = combos((".", *springs), tuple(sizes), curr_size, curr_target)
            else:
                c1 = 0
            # case: #
            if curr_size != curr_target:
                c2 = combos(tuple(springs), tuple(sizes), curr_size + 1, curr_target)
            else:
                c2 = 0
            return c1 + c2
    if curr_size != curr_target or sizes:
        return 0
    return 1


def b(data):
    total = 0
    for line in data.splitlines():
        springs, group_sizes = line.split(" ")
        springs = "?".join([springs] * 5)
        group_sizes = list(map(int, group_sizes.split(","))) * 5
        target = group_sizes.pop(0)
        total += combos(tuple(springs), tuple(group_sizes), 0, target)
    return total


examples = [
    "???.### 1,1,3",
    ".??..??...?##. 1,1,3",  # 4*8**4
    "?#?#?#?#?#?#?#? 1,3,1,6",
    "????.#...#... 4,1,1",
    "????.######..#####. 1,6,5",  # 4*5**4
    "?###???????? 3,2,1",  # Note: 10*15**4
]
example_answers_b = [1, 16384, 1, 16, 2500, 506250]
for example, example_answer_b in zip(examples, example_answers_b):
    example_answer = b(example)
    print(f"Example answer: {example_answer} (expecting: {example_answer_b})")
    assert example_answer == example_answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 60681419004564
puzzle.answer_b = answer
