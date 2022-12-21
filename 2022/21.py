import numpy as np
from aocd.models import Puzzle

YEAR = 2022
DAY = 21

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
d = {}
def a(data):
    for line in data.replace(":", "").splitlines():
        c = line.split(" ")
        try:
            value = int(c[1])
            assert len(c) == 2
            d[c[0]] = eval(f"lambda *_: {c[1]}")
        except ValueError:
            d[c[0]] = eval(f"lambda *_: d['{c[1]}']() {c[2]} d['{c[3]}']()")
    return int(d["root"]())

example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 152
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 121868120894282


# Part b
def b(data):
    exit()

example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 301
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
