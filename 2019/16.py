import numpy as np
from aocd.models import Puzzle

YEAR = 2019
DAY = 16

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, phases=100):
    inp = list(map(int, data))
    pattern = [0, 1, 0, -1]
    for _ in range(phases):
        out = []
        for e in range(1, len(inp) + 1):
            s = 0
            for i, d in enumerate(inp, 1):
                idx = (i % (len(pattern) * e)) // e
                s += d * pattern[idx]
            out.append(s)
        inp = [int(str(n)[-1]) for n in out]
    return int("".join(str(n) for n in inp)[:8])


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
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
