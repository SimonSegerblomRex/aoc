import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    grid = np.vstack([np.frombuffer(n.encode(), dtype=np.uint8) - ord("0") for n in data.split("\n")])
    height, width = grid.shape
    #nbr_edge = height * 2 + width * 2 - 4

    visible = np.full(grid.shape, False)
    visible[0, :] = True
    visible[-1, :] = True
    visible[:, 0] = True
    visible[:, -1] = True

    #top
    for i in range(1, height - 1):
        rows = grid[:i+1, :]
        visible[i, :] |= np.argmax(rows, axis=0) == i

    #left
    for j in range(1, width - 1):
        cols = grid[:, :j+1]
        visible[:, j] |= np.argmax(cols, axis=1) == j

    #bottom
    for i in range(1, height):
        rows = grid[i:, :]
        visible[i, :] |= np.argmax(rows[::-1, :], axis=0) == rows.shape[0] - 1

    #right
    for j in range(1, width):
        cols = grid[:, j:]
        visible[:, j] |= np.argmax(cols[:, ::-1], axis=1) == cols.shape[1] - 1

    return np.sum(visible)

example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 21
answer = a(puzzle.input_data)
print("a:", answer)
#puzzle.answer_a = answer


# Part b
def b(data):
    grid = np.vstack([np.frombuffer(n.encode(), dtype=np.uint8) - ord("0") for n in data.split("\n")])
    height, width = grid.shape
    #nbr_edge = height * 2 + width * 2 - 4

    grid = np.vstack([np.frombuffer(n.encode(), dtype=np.uint8) - ord("0") for n in data.split("\n")])
    height, width = grid.shape
    #nbr_edge = height * 2 + width * 2 - 4

    visible = np.full(grid.shape, False)
    visible[0, :] = True
    visible[-1, :] = True
    visible[:, 0] = True
    visible[:, -1] = True

    #top
    for i in range(1, height - 1):
        rows = grid[:i+1, :]
        visible[i, :] |= np.argmax(rows, axis=0) == i

    #left
    for j in range(1, width - 1):
        cols = grid[:, :j+1]
        visible[:, j] |= np.argmax(cols, axis=1) == j

    #bottom
    for i in range(1, height):
        rows = grid[i:, :]
        visible[i, :] |= np.argmax(rows[::-1, :], axis=0) == rows.shape[0] - 1

    #right
    for j in range(1, width):
        cols = grid[:, j:]
        visible[:, j] |= np.argmax(cols[:, ::-1], axis=1) == cols.shape[1] - 1

    """
    score = np.zeros(shape=grid.shape, like=grid)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            left = visible[i, :j].sum()
            right = visible[i, j+1:].sum()
            up = visible[:i, j].sum()
            down = visible[i+1:, j].sum()
            score[i,j] = up*down*left*right
    return score.max()
    """
    max_score = 0
    for i in range(height):
        for j in range(width):
            # right
            if j == width - 1:
                right = 0
            else:
                if (grid[i, :j] == grid[i, j +1]).all() and (grid[i, j + 1] < grid[i, j]):
                    right = width - j - 1
                    #breakpoint()
                else:
                    right = (grid[i, j + 1:].argmax() + 1) if (grid[i, j + 1] < grid[i, j]) else 1
            # left
            if j == 0:
                left = 0
            else:
                if (grid[i, :j] == grid[i, j -1]).all() and (grid[i, j - 1] < grid[i, j]):
                    left = j
                    #breakpoint()
                else:
                    left = (grid[i, :j][::-1].argmax() + 1) if (grid[i, j - 1] < grid[i, j]) else 1
            # up
            if i == 0:
                up = 0
            else:
                if (grid[:i, j] == grid[i - 1, j]).all() and (grid[i - 1, j] < grid[i, j]):
                    up = i
                    #breakpoint()
                else:
                    up = (grid[:i, j][::-1].argmax() + 1) if (grid[i - 1, j] < grid[i, j]) else 1
            # down
            if i == height - 1:
                down = 0
            else:
                if (grid[i, :j] == grid[i + 1, j]).all() and (grid[i + 1, j] < grid[i, j]):
                    down = height - i - 1
                    #breakpoint()
                else:
                    down = (grid[i + 1:, j].argmax() + 1) if (grid[i + 1, j] < grid[i, j]) else 1
            print(up, left, down, right, "a", i, j, "b", up*down*left*right)
            max_score = max(max_score, up * down * left * right)

    return max_score

example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 8
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
