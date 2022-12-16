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
    data = {e[0]: [int(e[1]), e[2].split(", ") or [e[2]]] for e in data}
    valves = list(data.keys())

    interesting_valves = [valve for valve, (p, _) in data.items() if p > 0]
    graph = {valve: data[valve][1] for valve in valves}
    dists = {}
    for start, end in itertools.combinations(interesting_valves, 2):
        dist = djikstra(start, end, graph)
        dists[(start, end)] = dist
        dists[(end, start)] = dist
    breakpoint()

    curr_pos = "AA"
    open_valves = []
    remaining_time = 30
    released_pressure = 0
    while remaining_time:
        curr_flow_rate, neighbours = data[curr_pos]
        if remaining_time == 1:
            # Only have option of opening valve at curr_pos
            released_pressure += data[curr_pos][0]
            break
        else:
            candidates = [valve for valve, (p, _) in data.items() if p > 0]
            # Calculate number of steps to each candidate
            # FIXME!
            breakpoint()
            options = [
                ((remaining_time - 1) * curr_flow_rate, curr_pos),
                *(((remaining_time - 2) * data[neigbour][0], neigbour) for neigbour in neighbours)
            ]
        pressure_to_be_released, next_move = sorted(options)[-1]
        if next_move == curr_pos:
            remaining_time -= 1
        else:
            # Assume that we always open the valve we move to...
            remaining_time -= 2
        print(next_move)
        released_pressure += pressure_to_be_released
        data[next_move][0] = 0
        curr_pos = next_move
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
