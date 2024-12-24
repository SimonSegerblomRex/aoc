import datetime
import re
from collections import defaultdict
from itertools import chain, combinations

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
    looking_for = set([g for g in values if g[0] == "z"])
    for gate in gates:
        gatess[gate[-1]] = f"values['{gate[0]}'] {opmap[gate[1]]} values['{gate[2]}']"
        if gate[-1][0] == "z":
            looking_for.add(gate[-1])
    while True:
        for g, f in gatess.items():
            try:
                values[g] = eval(f)
            except KeyError:
                pass
        if looking_for == looking_for & set(values):
            break
    out = []
    for k, v in values.items():
        if k[0] == "z":
            out.append((k, v))
    return int("0b" + "".join([str(b) for _, b in sorted(out)[::-1]]), 2)


answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 53755311654662


# Part b
def b(data, swap=2):
    initial_values, gates = data.split("\n\n")
    x = re.findall("x.+: (\d)", initial_values)
    y = re.findall("y.+: (\d)", initial_values)
    x = int("0b" + "".join(x[::-1]), 2)
    y = int("0b" + "".join(y[::-1]), 2)
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
    looking_for = set([g for g in values if g[0] == "z"])
    for gate in gates:
        gatess[gate[-1]] = f"values['{gate[0]}'] {opmap[gate[1]]} values['{gate[2]}']"
        if gate[-1][0] == "z":
            looking_for.add(gate[-1])
    values_orig = values.copy()
    hmm = combinations(list(gatess), 2)
    for combo in combinations(hmm, swap):
        values = values_orig.copy()
        for g0, g1 in combo:
            gatess[g0], gatess[g1] = gatess[g1], gatess[g0]
        while True:
            for g, f in gatess.items():
                try:
                    values[g] = eval(f)
                except KeyError:
                    pass
            if looking_for == looking_for & set(values):
                break
        out = []
        for k, v in values.items():
            if k[0] == "z":
                out.append((k, v))
        z = int("0b" + "".join([str(b) for _, b in sorted(out)[::-1]]), 2)
        if x + y == z:  # + for real input, & for example....
            break
        for g0, g1 in combo:
            gatess[g0], gatess[g1] = gatess[g1], gatess[g0]
    return ",".join(sorted(chain(*combo)))


example = """x00: 0
x01: 1
x02: 0
x03: 1
x04: 0
x05: 1
y00: 0
y01: 0
y02: 1
y03: 1
y04: 0
y05: 1

x00 AND y00 -> z05
x01 AND y01 -> z02
x02 AND y02 -> z01
x03 AND y03 -> z03
x04 AND y04 -> z04
x05 AND y05 -> z00"""
#answer = b(example, swap=2)
#print("example:", answer)
answer = b(puzzle.input_data, swap=4)
print("b:", answer)
#puzzle.answer_b = answer
