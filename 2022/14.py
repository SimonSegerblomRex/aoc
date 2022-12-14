import numpy as np
from aocd.models import Puzzle

YEAR = 2022
DAY = 14

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, floor=False):
    grid = np.zeros((1000, 1000), dtype=int)
    for line in data.splitlines():
        coords = line.split(" -> ")
        coords = [
            (int(coord.split(",")[1]), int(coord.split(",")[0])) for coord in coords
        ]
        for i in range(len(coords) - 1):
            i0, j0 = coords[i]
            i1, j1 = coords[i + 1]
            if j0 != j1:
                assert i0 == i1
                grid[i0, min(j0, j1) : max(j0, j1) + 1] = 1
            else:
                assert j0 == j1
                grid[min(i0, i1) : max(i0, i1) + 1, j0] = 1
    if floor:
        floor = np.argmax(grid == 1, axis=0).max() + 2
        grid[floor, :] = 1
        grid = grid[: floor + 1, :]
    else:
        abyss = np.argmax(grid == 1, axis=0).max()
        grid = grid[: abyss + 1, :]
    game_over = False
    counter = -1
    while not game_over:
        counter += 1
        i0 = 0
        j0 = 500
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
    return counter


example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 24
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 1298


# Part b
example_answer = a(puzzle.example_data, floor=True)
print(example_answer)
assert example_answer == 93
answer = a(puzzle.input_data, floor=True)
print("b:", answer)
assert answer == 25585
