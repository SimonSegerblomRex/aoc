import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    grid = np.zeros((1000, 1000), dtype=int)
    for line in data.splitlines():
        coords = line.split(" -> ")
        coords = [(int(coord.split(",")[1]), int(coord.split(",")[0])) for coord in coords]
        for i in range(len(coords) - 1):
            i0, j0 = coords[i]
            i1, j1 = coords[i + 1]
            if j0 != j1:
                assert i0 == i1
                grid[i0, min(j0, j1):max(j0, j1) + 1] = 1
            else:
                assert j0 == j1
                grid[min(i0, i1):max(i0, i1) + 1, j0] = 1
    #grid[0, 500] = 2
    print(grid[:10,494:504])
    abyss = np.argmax(grid==1, axis=0).max() # FIXME +1?
    grid = grid[:abyss+1, :]
    game_over = False
    counter = 0
    while not game_over:
        counter += 1
        i0 = 0
        j0 = 500
        print(i0, j0, counter)
        while True:
            if np.argmax(grid[i0:, j0]) == 0:
                game_over = True
                break
            i0 = i0 + np.argmax(grid[i0:, j0] != 0) - 1
            if grid[i0 + 1, j0 - 1] <= 0:
                i0 += 1
                j0 -= 1
            elif grid[i0 + 1, j0 + 1] <= 0:
                i0 += 1
                j0 += 1
            else:
                grid[i0, j0] = 2
                break
    return counter - 1

"""
example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 24
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer
"""

# Part b
def b(data):
    grid = np.zeros((1000, 1000), dtype=int)
    for line in data.splitlines():
        coords = line.split(" -> ")
        coords = [(int(coord.split(",")[1]), int(coord.split(",")[0])) for coord in coords]
        for i in range(len(coords) - 1):
            i0, j0 = coords[i]
            i1, j1 = coords[i + 1]
            if j0 != j1:
                assert i0 == i1
                grid[i0, min(j0, j1):max(j0, j1) + 1] = 1
            else:
                assert j0 == j1
                grid[min(i0, i1):max(i0, i1) + 1, j0] = 1
    #grid[0, 500] = 2
    floor = np.argmax(grid==1, axis=0).max() + 2
    grid[floor, :] = 1
    grid = grid[:floor + 1, :]
    print(grid[:12,494:504])
    breakpoint()
    game_over = False
    counter = 0
    while not game_over:
        counter += 1
        i0 = 0
        j0 = 500
        #print(i0, j0, counter)
        while True:
            if np.argmax(grid[i0:, j0]) == 0:
                game_over = True
                break
            i0 = i0 + np.argmax(grid[i0:, j0] != 0) - 1
            if grid[i0 + 1, j0 - 1] <= 0:
                i0 += 1
                j0 -= 1
            elif grid[i0 + 1, j0 + 1] <= 0:
                i0 += 1
                j0 += 1
            else:
                grid[i0, j0] = 2
                break
    return counter - 1

example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 93
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
