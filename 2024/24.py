import re
from itertools import combinations_with_replacement

from aocd.models import Puzzle

YEAR = 2024
DAY = 24

puzzle = Puzzle(year=YEAR, day=DAY)


def run(values, connections, looking_for):
    while True:
        old = values.copy()
        for g, f in connections.items():
            try:
                values[g] = eval(f)
            except KeyError:
                pass
        if values == old:
            break
        if looking_for == looking_for & set(values):
            break
    out = [(k, v) for k, v in values.items() if k[0] == "z"]
    if not out:
        return -1
    return int("0b" + "".join([str(b) for _, b in sorted(out)[::-1]]), 2)


def get_connections(data):
    initial_values, gates = data.split("\n\n")
    gates = re.findall("(.+) (.+) (.+) -> (.+)", gates)
    values = {}
    for value in initial_values.splitlines():
        gate, val = value.split(": ")
        values[gate] = int(val)
    opmap = {
        "AND": "&",
        "OR": "|",
        "XOR": "^",
    }
    looking_for = set([g for g in values if g[0] == "z"])
    connections = {}
    for in1, op, in2, out in gates:
        connections[out] = compile(f"values['{in1}'] {opmap[op]} values['{in2}']", "<string>", "eval")
        if out[0] == "z":
            looking_for.add(out)
    return values, connections, looking_for


# Part a
def a(data):
    values, connections, looking_for = get_connections(data)
    return run(values, connections, looking_for)


answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 53755311654662


def get_values(bits):
    numbers = [2**bits - 1, 1, 0]
    for x, y in combinations_with_replacement(numbers, 2):
        values = {}
        for bit in range(bits):
            values[f"x{bit:02}"] = (x >> bit) & 0b1
            values[f"y{bit:02}"] = (y >> bit) & 0b1
        yield x, y, values


# Part b
def b(data, swap=2):
    values, connections, looking_for = get_connections(data)
    bits = int(sorted(looking_for)[-1][1:3]) + 1
    looking_for = sorted(looking_for)
    gg = list(connections)
    out = []
    for bit in range(bits):
        ok = False
        i = 0
        j = 0
        while True:
            for x, y, values in get_values(bit + 1):
                z = run(values, connections, set(looking_for[:bit + 2]))
                mask = (1 << (bit + 1)) - 1
                if (x + y) & mask != z & mask:
                    break
            else:
                if i != 0 or j != 0:
                    out.append(i)
                    out.append(j)
                break
            # Something needs to be fixed...
            # Swap back
            connections[gg[i]], connections[gg[j]] = connections[gg[j]], connections[gg[i]]
            j += 1
            if j == len(connections):
                j = 0
                i += 1
            if i == len(connections):
                # Hopefully we don't end up here, or we need to
                # do more than one swap at the time...
                raise NotImplementedError("Can't handle more than one swap at the time...")
            # Swap
            connections[gg[i]], connections[gg[j]] = connections[gg[j]], connections[gg[i]]
        if len(out) == 8:
            break
    return ",".join(sorted([gg[g] for g in out]))


answer = b(puzzle.input_data, swap=4)
print("b:", answer)
assert answer == "dkr,ggk,hhh,htp,rhv,z05,z15,z20"
