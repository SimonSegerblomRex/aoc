import re

import numpy as np
from aocd.models import Puzzle

YEAR = 2023
DAY = 8

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def parse_data(data):
    lines = data.splitlines()
    instructions = lines[0]
    nodes = {}
    for match in re.finditer(
        r"(?P<node>\w+) = \((?P<left>\w+), (?P<right>\w+)\)", data
    ):
        nodes[match["node"]] = (match["left"], match["right"])
    instructions = [0 if instruction == "L" else 1 for instruction in instructions]
    return nodes, instructions


def nbr_of_steps(nodes, instructions, start_node, goal_nodes):
    i = 0
    n = len(instructions)
    curr_node = start_node
    s = 0
    while curr_node not in goal_nodes:
        instruction = instructions[i]
        i += 1
        i %= n
        s += 1
        curr_node = nodes[curr_node][instruction]
    return s


def a(data):
    nodes, instructions = parse_data(data)
    return nbr_of_steps(nodes, instructions, "AAA", "ZZZ")


answer = a(puzzle.input_data)
assert answer == 18157
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    nodes, instructions = parse_data(data)
    start_nodes = [node for node in nodes if node[-1] == "A"]
    goal_nodes = set([node for node in nodes if node[-1] == "Z"])
    steps = {
        start_node: nbr_of_steps(nodes, instructions, start_node, goal_nodes)
        for start_node in start_nodes
    }
    return np.lcm.reduce(list(steps.values()))


answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
assert answer == 14299763833181
