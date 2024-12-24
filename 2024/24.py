import datetime
import re
from collections import defaultdict
from itertools import chain, combinations, combinations_with_replacement

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


if 0:
    import schemdraw
    from schemdraw import logic
    from schemdraw.parsing import logicparse
    schemopmap = {
        "AND": logic.And,
        "OR":  logic.Or,
        "XOR": logic.Xor,
    }
    # https://schemdraw.readthedocs.io/en/latest/gallery/logicgate.html
    with schemdraw.Drawing() as d:
        d.config(unit=0.5)
        for in1, op, in2, out in gates:
            schemopmap[op]().label(out, "right")
        d.save("image.svg")
    #    d.add(logic.
    #    breakpoint()
    #d.save("image.svg")
    logicparse("(x0 & yo)")


def run(values, gatess, looking_for):
    while True:
        old = values.copy()
        for g, f in gatess.items():
            try:
                values[g] = eval(f)
            except KeyError:
                pass
        if values == old:
            break
        if looking_for == looking_for & set(values):
            break
    out = []
    for k, v in values.items():
        if k[0] == "z":
            out.append((k, v))
    if not out:
        return -1
    return int("0b" + "".join([str(b) for _, b in sorted(out)[::-1]]), 2)


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
        gatess[gate[-1]] = compile(f"values['{gate[0]}'] {opmap[gate[1]]} values['{gate[2]}']", "<string>", "eval")
        if gate[-1][0] == "z":
            looking_for.add(gate[-1])
    return run(values, gatess, looking_for)


answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 53755311654662


# Part b
def b(data, swap=2):
    initial_values, gates = data.split("\n\n")
    x = re.findall("x.+: (\d)", initial_values)
    y = re.findall("y.+: (\d)", initial_values)
    xx = int("0b" + "".join(x[::-1]), 2)
    yy = int("0b" + "".join(y[::-1]), 2)
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
    #for gate in gates:
    #    gatess[gate[-1]] = (gate[0], opmap[gate[1]], gate[2])
    #    if gate[-1][0] == "z":
    #        looking_for.add(gate[-1])
    for gate in gates:
        gatess[gate[-1]] = compile(f"values['{gate[0]}'] {opmap[gate[1]]} values['{gate[2]}']", "<string>", "eval")
        if gate[-1][0] == "z":
            looking_for.add(gate[-1])
    bits = int(sorted(looking_for)[-1][1:3]) + 1
    looking_for = sorted(looking_for)
    def get_values(bits):
        #numbers = range(2**bits - 1)
        numbers = [2**bits - 1, xx, yy, 0]#for bb in range(bits + 1)]
        #numbers = [2**bb - 1 for bb in range(bits + 1)]
        for x, y in combinations_with_replacement(numbers, 2):
            values = {}
            for bit in range(bits):
                values[f"x{bit:02}"] = (x >> bit) & 0b1
                values[f"y{bit:02}"] = (y >> bit) & 0b1
            yield x, y, values
    gg = list(gatess)
    gatess[gg[46]], gatess[gg[59]] = gatess[gg[59]], gatess[gg[46]]
    gatess[gg[7]], gatess[gg[100]] = gatess[gg[100]], gatess[gg[7]]
    gatess[gg[57]], gatess[gg[90]] = gatess[gg[90]], gatess[gg[57]]
    gatess[gg[71]], gatess[gg[195]] = gatess[gg[195]], gatess[gg[71]]
    return ",".join(sorted([gg[46], gg[59], gg[7], gg[100], gg[57], gg[90], gg[71], gg[195]]))
    for bit in range(36, bits):
        ok = False
        i = 0
        j = 0
        while True:
            for x, y, values in get_values(bit + 1):
                z = run(values, gatess, set(looking_for[:bit + 2]))
                mask = (1 << (bit + 1)) - 1
                if (x + y) & mask != z & mask:
                    break
            else:
                break
            # Something needs to be fixed...
            # Swap back
            gatess[gg[i]], gatess[gg[j]] = gatess[gg[j]], gatess[gg[i]]
            j += 1
            if j == len(gatess):
                j = 0
                i += 1
            if i == len(gatess):
                # Hopefully we don't end up here, or we need to
                # do more than one swap at the time...
                print("naj...")
                breakpoint()
            # Swap
            gatess[gg[i]], gatess[gg[j]] = gatess[gg[j]], gatess[gg[i]]
            print(bit, i, j)
        breakpoint()


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
puzzle.answer_b = answer
