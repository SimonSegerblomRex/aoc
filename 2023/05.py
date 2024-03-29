import re
from operator import attrgetter

import numpy as np
from aocd.models import Puzzle

YEAR = 2023
DAY = 5

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    lines = iter(data.splitlines())
    seeds = np.fromstring(next(lines).split(":")[1], dtype=int, sep=" ")
    big_map = []
    for match in re.finditer(
        r"(?P<from>\w+)-to-(?P<to>\w+) map:\n(?P<maps>[\d\s\n]+)", data
    ):
        maps = np.fromstring(match["maps"], dtype=int, sep=" ")
        maps.shape = (-1, 3)
        big_map.append(maps)
    ll = []
    for s in seeds:
        t = s
        for m in big_map:
            for r in m:
                if r[1] <= t < r[1] + r[2]:
                    t = r[0] + t - r[1]
                    break
        ll.append(t)
    return min(ll)


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 806029445


# Part b
def b(data):
    lines = iter(data.splitlines())
    seeds = np.fromstring(next(lines).split(":")[1], dtype=int, sep=" ")
    big_map = {}
    for match in re.finditer(
        r"(?P<from>\w+)-to-(?P<to>\w+) map:\n(?P<maps>[\d\s\n]+)", data
    ):
        maps = np.fromstring(match["maps"], dtype=int, sep=" ")
        maps.shape = (-1, 3)
        big_map[f"{match['from']}-{match['to']}"] = {
            "source_ranges": [range(r[1], r[1] + r[2]) for r in maps],
            "destination_ranges": [range(r[0], r[0] + r[2]) for r in maps],
        }
    seeds.shape = (-1, 2)
    seed_ranges = [range(r[0], r[0] + r[1]) for r in seeds]
    input_ranges = seed_ranges.copy()
    for step_name, step_ranges in big_map.items():
        mapped = []
        not_mapped = []
        for input_range in input_ranges:
            intersection_ranges = []
            for source_range, destination_range in zip(
                step_ranges["source_ranges"], step_ranges["destination_ranges"]
            ):
                intersection_range = range(
                    max(input_range.start, source_range.start),
                    min(input_range.stop, source_range.stop),
                )
                if intersection_range.stop > intersection_range.start:
                    intersection_ranges.append(intersection_range)
                    mapped.append(
                        range(
                            destination_range.start
                            + intersection_range.start
                            - source_range.start,
                            destination_range.start
                            + intersection_range.stop
                            - source_range.start,
                        )
                    )
            # Find ranges in input_range not mapped...
            curr_range = input_range
            for intersection_range in sorted(
                intersection_ranges, key=attrgetter("start")
            ):
                if curr_range.start < intersection_range.start:
                    not_mapped.append(range(curr_range.start, intersection_range.start))
                curr_range = range(intersection_range.stop, curr_range.stop)
            if curr_range.start < curr_range.stop:
                not_mapped.append(range(curr_range.start, curr_range.stop))
        input_ranges = mapped.copy()
        input_ranges.extend(not_mapped)
    return min(r.start for r in input_ranges)


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 59370572
