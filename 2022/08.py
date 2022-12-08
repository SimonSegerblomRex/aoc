import numpy as np
from aocd.models import Puzzle

YEAR = 2022
DAY = 8

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    grid = np.vstack(
        [np.frombuffer(n.encode(), dtype=np.uint8) - ord("0") for n in data.split("\n")]
    )
    height, width = grid.shape
    # nbr_edge = height * 2 + width * 2 - 4

    visible = np.full(grid.shape, False)
    visible[0, :] = True
    visible[-1, :] = True
    visible[:, 0] = True
    visible[:, -1] = True

    # top
    for i in range(1, height - 1):
        rows = grid[: i + 1, :]
        visible[i, :] |= np.argmax(rows, axis=0) == i

    # left
    for j in range(1, width - 1):
        cols = grid[:, : j + 1]
        visible[:, j] |= np.argmax(cols, axis=1) == j

    # bottom
    for i in range(1, height):
        rows = grid[i:, :]
        visible[i, :] |= np.argmax(rows[::-1, :], axis=0) == rows.shape[0] - 1

    # right
    for j in range(1, width):
        cols = grid[:, j:]
        visible[:, j] |= np.argmax(cols[:, ::-1], axis=1) == cols.shape[1] - 1

    return np.sum(visible)


example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 21
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 1688


# Part b
def b(data):
    grid = np.vstack(
        [np.frombuffer(n.encode(), dtype=np.uint8) - ord("0") for n in data.split("\n")]
    )
    height, width = grid.shape
    scores = np.zeros(grid.shape, dtype=int)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            right = grid[i, j:]
            left = grid[i, : j + 1][::-1]
            up = grid[: i + 1, j][::-1]
            down = grid[i:, j]

            def count(x):
                if (len(x) <= 2) or (x[0] < x[1]):
                    return 1
                c = 1
                for i in range(1, len(x) - 1):
                    if x[i] >= x[0]:
                        break
                    c += 1
                return c

            scores[i, j] = count(up) * count(down) * count(left) * count(right)
    return scores.max()


example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 8
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 410400
