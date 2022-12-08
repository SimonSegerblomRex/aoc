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
    scores = np.zeros(grid.shape, dtype=int)
    for i in range(1, height - 1):
        for j in range(1, width -1):
            right = grid[i, j:]
            left = grid[i, :j+1][::-1]
            up = grid[:i+1, j][::-1]
            down = grid[i:,j]

            def count(x):
                if (len(x) <= 2) or (x[0] < x[1]):
                    return 1
                c = 1
                for i in range(1, len(x) - 1):
                    if (x[i] >= x[0]):
                        break
                    c += 1
                return c
                breakpoint()

            scores[i,j] = count(up)*count(down)*count(left)*count(right)
            #breakpoint()
            """
            right = 0
            for x in range(j + 1, width):
                if (grid[i, x] <= grid[i, j]) and (grid[i, x] >= grid[i, x - 1]):
                    right +=1
                else:
                    if x < width - 1:
                        right += 1
                    break

            left = 0
            for x in range(j - 1, -1, -1):
                if (grid[i, x] < grid[i, j]) and (grid[i,x] >= grid[i, x + 1]):
                    left +=1
                else:
                    if x > 0:
                        left += 1
                    break

            down = 0
            for y in range(i + 1, height):
                if (grid[y, j] <= grid[i, j]) and (grid[y,j] >= grid[y -1,j]):
                    down +=1
                else:
                    if y < height -1:
                        down += 1
                    break

            up = 0
            for y in range(i - 1, -1, -1):
                if (grid[y, j] <= grid[i, j]) and (grid[y,j] >= grid[y + 1,j]):
                    up +=1
                else:
                    if y > 0:
                        up += 1
                    break

            #scores[i,j] = min(up, 1) * min(down, 1) * min(left, 1) * min(right, 1)
            scores[i,j] = up * down * left * right
            """

            """
            # right
            if j == width - 1:
                right = 0
            else:
                right = (grid[i, j + 1:].argmax() + 1) if (grid[i, j + 1] <= grid[i, j]) else 1
            # left
            if j == 0:
                left = 0
            else:
                left = (grid[i, :j].argmax()) if (grid[i, j - 1] <= grid[i, j]) else 1
            # up
            if i == 0:
                up = 0
            else:
                up = (grid[:i, j][::-1].argmax() + 1) if (grid[i - 1, j] <= grid[i, j]) else 1
            # down
            if i == height - 1:
                down = 0
            else:
                down = (grid[i + 1:, j].argmax() + 1) if (grid[i + 1, j] <= grid[i, j]) else 1
            """
            #max_score = max(max_score, up * down * left * right)

    #breakpoint()
    print(scores)
    return scores.max()#max_score

example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 8
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
