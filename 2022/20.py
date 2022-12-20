import numpy as np
from aocd.models import Puzzle

YEAR = 2022
DAY = 20

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def mix(data, order):
    data = np.array(data).tolist()
    old_pos = 0
    l = len(data)
    for i in range(l):
        old_pos = order.index(i)
        v = data.pop(old_pos)
        o = order.pop(old_pos)
        new_pos = (old_pos + v) % (l - 1)
        if new_pos == 0:
            new_pos = l - 1
        data.insert(new_pos, v)
        order.insert(new_pos, o)
        if 0:
            print(v, old_pos, new_pos)
            print(data)
            breakpoint()
    return data, order


def a(data):
    data = np.fromstring(data, dtype=int, sep="\n")
    order = list(range(len(data)))
    data, _ = mix(data, order)
    i0 = data.index(0)
    l = len(data)
    return data[(i0 + 1000) % l] + data[(i0 + 2000) % l] + data[(i0 + 3000) % l]


example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 3
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    data = np.fromstring(data, dtype=int, sep="\n")
    data *= 811589153
    order = list(range(len(data)))
    for _ in range(10):
        data, order = mix(data, order)
    i0 = data.index(0)
    l = len(data)
    return data[(i0 + 1000) % l] + data[(i0 + 2000) % l] + data[(i0 + 3000) % l]


example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 1623178306
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
