from ast import literal_eval

import numpy as np
from aocd.models import Puzzle

YEAR = 2019
DAY = 12

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, steps=1000):
    data = data.replace("<", "(").replace(">", ")").replace("=", "")
    data = data.replace("x", "").replace("y", "").replace("z", "")
    pos = []
    for line in data.splitlines():
        pos.append(literal_eval(line))
    pos = np.array(pos)
    vel = np.zeros(shape=pos.shape, dtype=int)
    for _ in range(steps):
        for i in range(len(pos)):
            vel[i] += (pos > pos[i]).sum(axis=0)
            vel[i] -= (pos < pos[i]).sum(axis=0)
        pos += vel
    pot = np.abs(pos).sum(axis=1)
    kin = np.abs(vel).sum(axis=1)
    tot = pot * kin
    return tot.sum()


answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    print(data)
    breakpoint()


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
