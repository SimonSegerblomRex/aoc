import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 7

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    positions = np.fromstring(data, dtype=int, sep=",")
    costs = []
    for alignment_pos in range(positions.max() + 1):
        costs.append(np.sum(np.abs(positions - alignment_pos)))
    return np.min(costs)


assert a(puzzle.example_data) == 37
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 356992


# Part b
def b(data):
    positions = np.fromstring(data, dtype=int, sep=",")
    costs = []
    for alignment_pos in range(positions.max() + 1):
        steps = np.abs(positions - alignment_pos)
        costs.append(((steps + 1) / 2 * steps).astype(int).sum())
    return np.min(costs)


assert b(puzzle.example_data) == 168
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 101268110
