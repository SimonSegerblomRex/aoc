import numpy as np
from aocd.models import Puzzle

puzzle = Puzzle(year=2021, day=1)

def a(input):
    input = np.fromstring(input, dtype=int, sep="\n")
    diff = np.diff(input)
    return np.sum(diff > 0)

assert a(puzzle.example_data) == 7
#puzzle.answer_a = a(puzzle.input_data)

def b(input):
    input = np.fromstring(input, dtype=int, sep="\n")
    w = np.lib.stride_tricks.sliding_window_view(input, window_shape=3)
    diff = np.diff(np.sum(w, axis=1))
    return np.sum(diff > 0)

assert b(puzzle.example_data) == 5
puzzle.answer_b = b(puzzle.input_data)
