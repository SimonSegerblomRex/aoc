import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)

PATTERN = r"Valve ([A-Z]+) has flow rate=(\d+); tunnels? leads? to valves? (.+)"

# Part a
def a(data):
    data = re.findall(PATTERN, data)
    data = {e[0]: [int(e[1]), e[2].split(", ") or [e[2]]] for e in data}
    valves = list(data.keys())
    curr_pos = "AA"
    open_valves = []
    remaining_time = 30
    released_pressure = 0
    while remaining_time:
        curr_flow_rate, neighbours = data[curr_pos]
        if remaining_time == 1:
            # Only have option of opening valve at curr_pos
            next_move == curr_pos
        else:
            ...
        # FIXME! Calculate pressure release potential for all possible destinations!

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
        data[curr_pos][0] = 0
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
