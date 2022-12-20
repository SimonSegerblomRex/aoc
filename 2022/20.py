import numpy as np
from aocd.models import Puzzle

YEAR = 2022
DAY = 20

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def mix(data):
    data = np.array(data).tolist()
    moved = np.full(len(data), False)
    moved = moved.tolist()
    old_pos = 0
    l = len(data)
    for i in range(l):
        old_pos = np.argmin(moved)
        v = data[old_pos]
        new_pos = (old_pos + v) % (l - 1)
        if new_pos == 0:
            new_pos = l - 1
        moved.pop(old_pos)
        data.pop(old_pos)
        data.insert(new_pos, v)
        moved.insert(new_pos, True)
        if 0:
            print(v, old_pos, new_pos)
            print(data)
            breakpoint()
    i0 = data.index(0)
    return data[(i0 + 1000) % l] + data[(i0 + 2000) % l] + data[(i0 + 3000) % l]


def a(data):
    data = np.fromstring(data, dtype=int, sep="\n")
    return mix(data)


example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 3
breakpoint()
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    exit()


example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 3
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
