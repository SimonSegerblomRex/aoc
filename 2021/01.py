import numpy as np
from aocd.models import Puzzle

puzzle = Puzzle(year=2021, day=1)


def a(data):
    data = np.fromstring(data, dtype=int, sep="\n")
    diff = np.diff(data)
    return np.sum(diff > 0)


assert a(puzzle.example_data) == 7
answer = a(puzzle.input_data)
print("a:", answer)


def b(data):
    data = np.fromstring(data, dtype=int, sep="\n")
    w = np.lib.stride_tricks.sliding_window_view(data, window_shape=3)
    diff = np.diff(np.sum(w, axis=1))
    return np.sum(diff > 0)


assert b(puzzle.example_data) == 5
answer = b(puzzle.input_data)
print("b:", answer)
