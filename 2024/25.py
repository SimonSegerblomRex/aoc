from aocd.models import Puzzle

YEAR = 2024
DAY = 25

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    schematics = data.split("\n\n")
    locks = []
    keys = []
    for schematic in schematics:
        rows = [-1] * 5
        for line in schematic.split():
            for j, c in enumerate(line):
                if c == "#":
                    rows[j] += 1
        if schematic[0] == "#":
            locks.append(rows)
        else:
            keys.append(rows)
    s = 0
    for lock in locks:
        for key in keys:
            for col in range(5):
                if lock[col] + key[col] > 5:
                    break
            else:
                s += 1
    return s


answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 2993
