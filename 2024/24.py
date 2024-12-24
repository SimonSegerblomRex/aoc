import datetime
import re
from collections import defaultdict

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    initial_values, gates = data.split("\n\n")
    gates = re.findall("(.+) (.+) (.+) -> (.+)", gates)
    values = {}
    for value in initial_values.splitlines():
        tmp = value.split(": ")
        values[tmp[0]] = int(tmp[1])
    gatess = {}
    opmap = {
        "AND": "&",
        "OR": "|",
        "XOR": "^",
    }
    for gate in gates:
        gatess[gate[-1]] = f"values['{gate[0]}'] {opmap[gate[1]]} values['{gate[2]}']"
    done = False
    i = 0
    while not done:
        if i > 1e4:
            break
        i += 1
        for g, f in gatess.items():
            try:
                values[g] = eval(f)
            except KeyError:
                pass
    out = []
    for k, v in values.items():
        if k[0] == "z":
            out.append((k, v))
    return int("0b" + "".join([str(b) for _, b in sorted(out)[::-1]]), 2)


answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    breakpoint()


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
