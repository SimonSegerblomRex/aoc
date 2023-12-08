import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    lines = data.splitlines()
    instructions = lines[0]
    nodes = {}
    for match in re.finditer(
        r"(?P<node>\w+) = \((?P<left>\w+), (?P<right>\w+)\)", data
    ):
        nodes[match["node"]] = (match["left"], match["right"])
    i = 0
    n = len(instructions)
    instructions = [0 if instruction == "L" else 1 for instruction in instructions]
    curr_node = "AAA"
    s = 0
    while curr_node != "ZZZ":
        instruction = instructions[i]
        i += 1
        i %= n
        s += 1
        curr_node = nodes[curr_node][instruction]
    return s


answer = a(puzzle.input_data)
assert answer == 18157
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    lines = data.splitlines()
    instructions = lines[0]
    nodes = {}
    for match in re.finditer(
        r"(?P<node>\w+) = \((?P<left>\w+), (?P<right>\w+)\)", data
    ):
        nodes[match["node"]] = (match["left"], match["right"])
    i = 0
    n = len(instructions)
    instructions = [0 if instruction == "L" else 1 for instruction in instructions]
    curr_nodes = [node for node in nodes if node[-1] == "A"]
    goal_nodes = set([node for node in nodes if node[-1] == "Z"])
    s = 0
    while set(curr_nodes) != goal_nodes:
        instruction = instructions[i]
        i += 1
        i %= n
        curr_nodes = [nodes[curr_node][instruction] for curr_node in curr_nodes]
        s += 1
    return s


example = """LR

11A = (11B, XXX)
11B = (XXX, 11Z)
11Z = (11B, XXX)
22A = (22B, XXX)
22B = (22C, 22C)
22C = (22Z, 22Z)
22Z = (22B, 22B)
XXX = (XXX, XXX)"""

example_answer = b(example)
print("example:", example_answer)
assert example_answer == 6

answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
