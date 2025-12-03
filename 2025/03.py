import numpy as np
from aocd.models import Puzzle

YEAR = 2025
DAY = 3

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, d=2):
    s = 0
    for line in data.split():
        dj = ["0"] * d
        l = len(line)
        i = 0
        for j in range(d):
            dj[j] = max(line[i : l - d + j + 1])
            i += np.argmax(list(line[i : l - d + j + 1])) + 1
        s += int("".join(dj))
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 16858


# Part b
for example in puzzle.examples:
    if example.answer_b:
        example_answer = a(example.input_data, 12)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = a(puzzle.input_data, 12)
print("b:", answer)
assert answer == 167549941654721
