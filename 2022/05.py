import re

from aocd.models import Puzzle

YEAR = 2022
DAY = 5

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def _parse_crates(data):
    ugly_crates = re.findall(r"(.{4})", data[: data.find("1")], re.DOTALL)
    nbr_crates = int(re.findall(r"\d+\s+(\d+) \n", data)[0])
    crates = {}
    for i in range(nbr_crates):
        crates[i + 1] = [
            e.strip(" \n[]") for e in ugly_crates[i::nbr_crates] if e.strip(" \n[]")
        ]
    return crates


def a(data):
    crates = _parse_crates(data)
    instructions = re.findall(r"move (\d+) from (\d+) to (\d+)", data)
    for instruction in instructions:
        nbr, fr, to = map(int, instruction)
        for _ in range(nbr):
            box = crates[fr].pop(0)
            crates[to].insert(0, box)
    return "".join(crate[0] for crate in crates.values())


example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == "CMZ"
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == "CNSZFDVLJ"


# Part b
def b(data):
    crates = _parse_crates(data)
    instructions = re.findall(r"move (\d+) from (\d+) to (\d+)", data)
    for instruction in instructions:
        nbr, fr, to = map(int, instruction)
        for i in range(nbr):
            box = crates[fr].pop(0)
            crates[to].insert(i, box)
    return "".join(crate[0] for crate in crates.values())


example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == "MCD"
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == "QNDWLMGNS"
