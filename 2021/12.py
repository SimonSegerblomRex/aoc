from collections import defaultdict
from itertools import chain

import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 12

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    data = data.splitlines()
    connections = [c.split("-") for c in data]
    caves = set(chain.from_iterable(connections))
    cave_map = defaultdict(set)
    for cave in caves:
        for connection in connections:
            cave_map[connection[0]].add(connection[1])
            cave_map[connection[1]].add(connection[0])
    paths = []

    def find_paths(path):
        curr_cave = path[-1]
        for next_cave in cave_map[curr_cave]:
            if next_cave.islower() and next_cave in path:
                continue
            new_path = [*path, next_cave]
            if next_cave == "end":
                paths.append(new_path)
            else:
                find_paths(new_path)
        return paths

    find_paths(["start"])
    return len(paths)


example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 10
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 4167

# Part b
def b(data):
    data = data.splitlines()
    connections = [c.split("-") for c in data]
    caves = set(chain.from_iterable(connections))
    cave_map = defaultdict(set)
    for cave in caves:
        for connection in connections:
            cave_map[connection[0]].add(connection[1])
            cave_map[connection[1]].add(connection[0])
    paths = []
    small_caves = [
        cave for cave in caves if cave.islower() and cave not in ["start", "end"]
    ]

    def find_paths(path):
        curr_cave = path[-1]
        for next_cave in cave_map[curr_cave]:
            if next_cave == "start":
                continue
            new_path = [*path, next_cave]
            if next_cave in small_caves:
                small_visit_counts = [new_path.count(c) for c in small_caves]
                if (small_visit_counts.count(2) > 1) or max(small_visit_counts) > 2:
                    continue
            if next_cave == "end":
                paths.append(new_path)
            else:
                find_paths(new_path)
        return paths

    find_paths(["start"])
    return len(paths)


example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 36
answer = b(puzzle.input_data)
print("b:", answer)
# puzzle.answer_b = answer
