import numpy as np
from aocd.models import Puzzle

YEAR = 2023
DAY = 9

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def extrapolate(sequence):
    diff = np.diff(sequence)
    if diff.any():
        return sequence[-1] + extrapolate(diff)
    return sequence[-1]

def a(data):
    s = 0
    for line in data.splitlines():
        sequence = np.fromstring(line, sep=" ", dtype=int)
        s += extrapolate(sequence)
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 1772145754


# Part b
def b(data):
    s = 0
    for line in data.splitlines():
        sequence = np.fromstring(line, sep=" ", dtype=int)[::-1]
        s += extrapolate(sequence)
    return s


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 867
