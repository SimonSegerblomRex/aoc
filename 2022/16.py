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
        for i, next_pos in enumerate(possible_next_pos):
            if time_remaining - (dists[curr_pos][next_pos] + 1) <= 0:
                paths.append(path.copy())
                continue
            find_paths(
                [*path, next_pos],
                time_remaining - (dists[curr_pos][next_pos] + 1)
            )

    find_paths(["AA"], time_remaining=30)
    breakpoint()

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

    #breakpoint()
    print(max_pressure_released)
    breakpoint()
    return max_pressure_released

    # hmmm
    curr_pos = "AA"
    open_valves = []
    remaining_time = 30
    released_pressure = 0
    while remaining_time:
        curr_flow_rate, neighbours = data[curr_pos]
        other_candidates = neighbours.intersection(interesting_valves)  # FIXME: Should we do this..?
        if 0:
            options = [
                #(pressure_to_be_released, node, nbr_of_steps)
                ((remaining_time - 1) * curr_flow_rate, curr_pos, 0),
                *(((remaining_time - dists[curr_pos][valve] - 1) * data[valve][0], valve, dists[curr_pos][valve]) for valve in other_candidates)
            ]
        else:
            options = [
                ((remaining_time - dists[curr_pos][valve] - 1) * data[valve][0], valve, dists[curr_pos][valve]) for valve in other_candidates
            ]
        options = [option for option in options if ((option[-1] + 1) <= remaining_time)]
        if not options:
            break
        pressure_to_be_released, next_pos, nbr_of_steps = sorted(options)[-1]
        print(curr_pos, next_pos)
        released_pressure += pressure_to_be_released
        curr_pos = next_pos
        remaining_time -= nbr_of_steps + 1
        interesting_valves.remove(next_pos)
    return released_pressure


example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 1651
answer = a(puzzle.input_data)
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
