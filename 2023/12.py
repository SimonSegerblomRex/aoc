import datetime
import re
import itertools

import numpy as np
from aocd.models import Puzzle
from more_itertools import distinct_permutations

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

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
def b_old(data):
    total = 0
    for i, line in enumerate(data.splitlines()):
        print(i)
        c = 1
        springs, group_sizes = line.split(" ")
        springs = "?".join([springs] * 5)
        group_sizes = ",".join([group_sizes] * 5)
        group_sizes = np.fromstring(group_sizes, sep=",", dtype=int).tolist()
        #print(springs, group_sizes)
        for group in springs.split("."):
            print(group, group_sizes)
            if not group:
                continue
            if len(group) < group_sizes[0]:
                continue
            size = 0
            subgroup_sizes = []
            while True:
                if size + group_sizes[0] > len(group):
                    break
                subgroup_size = group_sizes.pop(0)
                subgroup_sizes.append(subgroup_size)
                size += subgroup_size
                if size + 1 >= len(group):
                    break
                if not group_sizes:
                    break
                size += 1
            if not group_sizes:
                break
            print("!!!")
            c *= a(f"{group} {','.join(str(i) for i in subgroup_sizes)}")
        total += c
    return total


def combos(springs, sizes, curr_size, curr_target):
    for i, c in enumerate(springs):
        if c == ".":
            if curr_size:
                if curr_size == curr_target:
                    if not sizes:
                        break
                    curr_target = sizes.pop(0)
                    curr_size = 0
                else:
                    return 0
        elif c == "#":
            curr_size += 1
        elif c == "?":
            # case .: combos("." + springs[i + 1:], sizes, curr_size)
            # case #: combos(springs[i + 1:], sizes, curr_size + 1)
            c1 = combos("." + springs[i + 1:], sizes, curr_size, curr_target)
            c2 = combos(springs[i + 1:], sizes, curr_size + 1, curr_target)
            print(c1, c2)
            return c1 + c2
            return combos("." + springs[i + 1:], sizes, curr_size, curr_target) + combos(springs[i + 1:], sizes, curr_size + 1, curr_target)
    if sizes:
        return 0
    if curr_size != curr_target:
        return 0
    if "?" in springs:
        return 0
    return 1


def b(data):
    total = 0
    for i, line in enumerate(data.splitlines()):
        springs, group_sizes = line.split(" ")
        springs = "?".join([springs] * 5)
        group_sizes = ",".join([group_sizes] * 5)
        group_sizes = np.fromstring(group_sizes, sep=",", dtype=int).tolist()
        print("AA:", springs, group_sizes)
        total += combos(springs, group_sizes, 0, group_sizes[0])
    return total


examples = [
    "???.### 1,1,3",
    ".??..??...?##. 1,1,3",  # 4*8**4
    "?#?#?#?#?#?#?#? 1,3,1,6",
    "????.#...#... 4,1,1",
    "????.######..#####. 1,6,5", # 4*5**4
    "?###???????? 3,2,1",  # Note: 10*15**4
]
example_answers_b = [1, 16384, 1, 16, 2500, 506250]
for example, example_answer_b in zip(examples, example_answers_b):
    example_answer = b(example)
    print(f"Example answer: {example_answer} (expecting: {example_answer_b})")
    assert example_answer == example_answer_b
answer = b(puzzle.input_data)
print("b:", answer)
# puzzle.answer_b = answer
