import re

from aocd.models import Puzzle

YEAR = 2024
DAY = 3

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    tmp = re.findall("mul\((\d+),(\d+)\)", data)
    s = 0
    for t in tmp:
        s += int(t[0]) * int(t[1])
    return s


answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 157621318

# Part b
def b(data):
    pause = False
    s = 0
    for m in re.finditer("mul\((\d+),(\d+)\)|(do\(\))|(don't\(\))", data):
        if m.group(0) == "don't()":
            pause = True
            continue
        if m.group(0) == "do()":
            pause = False
            continue
        if pause:
            continue
        s += int(m.group(1)) * int(m.group(2))
    return s


answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 79845780
