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
    zero_rows = []
    for i, row in enumerate(range(grid.shape[0])):
        row = grid[i, :]
        if row.any():
            continue
        zero_rows.append(i)
    zero_cols = []
    for j, col in enumerate(range(grid.shape[1])):
        col = grid[:, j]
        if col.any():
            continue
        zero_cols.append(j)
    galaxies = np.nonzero(grid)
    from collections import defaultdict

    distances = defaultdict(lambda: np.inf)
    for i, galaxy_from in enumerate(zip(*galaxies)):
        for j, galaxy_to in enumerate(zip(*galaxies)):
            i_min = min(galaxy_from[0], galaxy_to[0])
            i_max = max(galaxy_from[0], galaxy_to[0])
            extra_i = 0
            for r in zero_rows:
                if i_min < r < i_max:
                    extra_i += extra
            j_min = min(galaxy_from[1], galaxy_to[1])
            j_max = max(galaxy_from[1], galaxy_to[1])
            extra_j = 0
            for c in zero_cols:
                if j_min < c < j_max:
                    extra_j += extra
            d = i_max + extra_i - i_min + j_max - j_min + extra_j
            if d:
                idx = tuple(sorted((i, j)))
                distances[idx] = min(distances[idx], d)
    return sum(distances.values())


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
