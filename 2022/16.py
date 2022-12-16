import datetime
import itertools
import re
from collections import defaultdict

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)

PATTERN = r"Valve ([A-Z]+) has flow rate=(\d+); tunnels? leads? to valves? (.+)"


# Part a
def djikstra(start, end, graph):
    steps = 0
    dist = defaultdict(lambda: np.inf)
    prev = defaultdict(lambda: None)
    dist[start] = 0

    to_visit = [start]
    while to_visit:
        curr_node = to_visit.pop(0)
        if curr_node == end:
            break
        for node in graph[curr_node]:
            tentative_dist = dist[curr_node] + 1
            if tentative_dist < dist[node]:
                dist[node] = tentative_dist
                prev[node] = curr_node
                if node not in to_visit:
                    to_visit.append(node)

    return dist[end]


def a(data):
    data = re.findall(PATTERN, data)
    data = {e[0]: [int(e[1]), set(e[2].split(", ")) or [e[2]]] for e in data}
    valves = list(data.keys())

    interesting_valves = set(valve for valve, (p, _) in data.items() if p > 0)
    graph = {valve: data[valve][1] for valve in valves}
    dists = defaultdict(dict)
    for start, end in itertools.combinations(valves, 2):
        dist = djikstra(start, end, graph)
        dists[start][start] = 0
        dists[end][end] = 0
        dists[start][end] = dist
        dists[end][start] = dist

    # brute-force
    paths = []
    def find_paths(path, time_remaining):
        curr_pos = path[-1]
        possible_next_pos = interesting_valves - set(path)
        if not possible_next_pos or (time_remaining <= 0):
            paths.append(path.copy())
            return
        while possible_next_pos:
            next_pos = possible_next_pos.pop()
            next_time_remaining = time_remaining - (dists[curr_pos][next_pos] + 1)
            if next_time_remaining <= 0:
                paths.append(path.copy())
                continue
            find_paths(
                [*path, next_pos],
                next_time_remaining,
            )

    find_paths(["AA"], time_remaining=30)

    max_pressure_released = 0
    for path in paths:
        if not path:
            continue
        time_remaining = 30
        pressure_released = 0
        curr_pos = path.pop(0)
        for next_pos in path:
            time_remaining -= dists[curr_pos][next_pos] + 1
            if time_remaining <= 0:
                break
            pressure_released += time_remaining * data[next_pos][0]
            curr_pos = next_pos
        max_pressure_released = max(max_pressure_released, pressure_released)

    return max_pressure_released


example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 1651
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 1880


# Part b
def b(data):
    data = re.findall(PATTERN, data)
    data = {e[0]: [int(e[1]), set(e[2].split(", ")) or [e[2]]] for e in data}
    valves = list(data.keys())

    interesting_valves = set(valve for valve, (p, _) in data.items() if p > 0)
    graph = {valve: data[valve][1] for valve in valves}
    dists = defaultdict(dict)
    for start, end in itertools.combinations(valves, 2):
        dist = djikstra(start, end, graph)
        dists[start][start] = 0
        dists[end][end] = 0
        dists[start][end] = dist
        dists[end][start] = dist

    # brute-force
    your_paths = []
    elephant_paths = []
    def find_paths(your_path, elephant_path, your_time_remaining, elephant_time_remaining):
        your_curr_pos = your_path[-1]
        elephant_curr_pos = elephant_path[-1]
        possible_next_pos = interesting_valves - set(your_path) - set(elephant_path)
        if not possible_next_pos or (your_time_remaining <= 0):
            your_paths.append(your_path.copy())
            return
        while possible_next_pos:
            your_next_pos = possible_next_pos.pop()
            elephant_next_pos = possible_next_pos.pop()
            your_next_time_remaining = time_remaining - (dists[curr_pos][your_next_pos] + 1)
            elephant_next_time_remaining = time_remaining - (dists[curr_pos][elephant_next_pos] + 1)
            if your_next_time_remaining <= 0:
                paths.append(path.copy())
                continue
            find_paths(
                [*path, next_pos],
                next_time_remaining,
            )

            if time_remaining - (dists[curr_pos][next_pos] + 1) <= 0:
                paths.append(path.copy())
                continue
            find_paths(
                [*path, next_pos],
                time_remaining - (dists[curr_pos][next_pos] + 1)
            )

    find_paths(["AA"], ["AA"], your_time_remaining=26, elephant_time_remaining=26)

    max_pressure_released = 0
    for path in paths:
        if not path:
            continue
        time_remaining = 26
        pressure_released = 0
        curr_pos = path.pop(0)
        for next_pos in path:
            time_remaining -= dists[curr_pos][next_pos] + 1
            if time_remaining <= 0:
                break
            pressure_released += time_remaining * data[next_pos][0]
            curr_pos = next_pos
        max_pressure_released = max(max_pressure_released, pressure_released)

    return max_pressure_released


example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 1707
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
