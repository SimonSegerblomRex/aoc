import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    file_size = list(map(int, data[::2]))
    free_space = list(map(int, data[1::2]))
    free_space.append(0)
    file_id = list(range(len(file_size)))
    uncompressed_size = sum(file_size) + sum(free_space)
    compressed_size = sum(file_size)
    m = np.full(uncompressed_size, -1, dtype=int)
    p = 0
    for f_id, f_size, space in zip(file_id, file_size, free_space):
        m[p:p + f_size] = f_id
        p += f_size
        p += space
    p = 0
    for p in range(uncompressed_size):
        if m[p] > -1:
            continue
        w = np.where(m > -1)[0]
        l =  w[-1]
        if l < p:
            break
        m[p] = m[l]
        m[l] = -1
    m = m[:p]
    s = 0
    for i, n in enumerate(m):
        s += i * n
    return s


if 1:
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
