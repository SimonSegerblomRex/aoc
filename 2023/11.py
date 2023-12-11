from collections import defaultdict
from itertools import combinations

import numpy as np
from aocd.models import Puzzle

YEAR = 2023
DAY = 11

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, extra):
    grid = np.vstack(
        [np.frombuffer(row.encode(), dtype=np.uint8) for row in data.splitlines()]
    )
    grid[grid == ord(".")] = 0
    grid[grid == ord("#")] = 1
    zero_rows = np.flatnonzero(np.all(grid == 0, axis=1))
    zero_cols = np.flatnonzero(np.all(grid == 0, axis=0))
    galaxies = np.nonzero(grid)
    distances = []
    for galaxy_from, galaxy_to in combinations(zip(*galaxies), 2):
        i_min, i_max = sorted((galaxy_from[0], galaxy_to[0]))
        extra_i = (
            np.count_nonzero((i_min < zero_rows) & (zero_rows < i_max)) * extra
        )
        j_min, j_max = sorted((galaxy_from[1], galaxy_to[1]))
        extra_j = (
            np.count_nonzero((j_min < zero_cols) & (zero_cols < j_max)) * extra
        )
        d = i_max + extra_i - i_min + j_max - j_min + extra_j
        distances.append(d)
    return sum(distances)


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data, 1)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data, 1)
print("a:", answer)
assert answer == 9681886

# Part b
example_answer = a(example.input_data, 10 - 1)
assert example_answer == 1030

example_answer = a(example.input_data, 100 - 1)
assert example_answer == 8410

answer = a(puzzle.input_data, 1000000 - 1)
print("b:", answer)
assert answer != 791134890760
puzzle.answer_b = answer
