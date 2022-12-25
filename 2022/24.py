import copy

import numpy as np
from aocd.models import Puzzle

YEAR = 2022
DAY = 24

puzzle = Puzzle(year=YEAR, day=DAY)

EXAMPLE_DATA = """#.######
#>>.<^<#
#.<..<<#
#>v.><>#
#<^v^^>#
######.#"""

# Part a
def debug_print(walls, winds, curr_pos):
    width, height = np.max(walls, axis=1) + 1
    print("")
    for y in range(height):
        for x in range(width):
            if np.all(walls == [[x],[y]], axis=0).any():
                print("#", end="")
            elif np.all(winds["u"] == [[x],[y]], axis=0).any():
                print("^", end="")
            elif np.all(winds["d"] == [[x],[y]], axis=0).any():
                print("v", end="")
            elif np.all(winds["l"] == [[x],[y]], axis=0).any():
                print("<", end="")
            elif np.all(winds["r"] == [[x],[y]], axis=0).any():
                print(">", end="")
            elif (x, y) == curr_pos:
                print("E")
            else:
                #FIXME
                print(".", end="")
        print("")
    breakpoint()

def a(data):
    rows = [
        np.frombuffer(row.encode(), dtype=np.uint8)
        for row in data.splitlines()
    ]
    grid = np.vstack(rows)
    height, width = grid.shape
    start_y = 0
    start_x = np.flatnonzero(grid[start_y] == ord("."))[0]
    start = (start_x, start_y)
    goal_y = height - 1
    goal_x = np.flatnonzero(grid[goal_y] == ord("."))[0]
    goal = (goal_x, goal_y)
    walls = np.vstack(np.nonzero(grid == ord("#")))[::-1]
    winds = {
        "u": np.vstack(np.nonzero(grid == ord("^")))[::-1],
        "d": np.vstack(np.nonzero(grid == ord("v")))[::-1],
        "l": np.vstack(np.nonzero(grid == ord("<")))[::-1],
        "r": np.vstack(np.nonzero(grid == ord(">")))[::-1],
    }

    wind_states = np.lcm(height - 2, width - 2)

    grid3D = np.zeros((width, height, wind_states), dtype=np.uint8)
    grid3D[walls[0, :], walls[1, :], :] = 3
    for i in range(wind_states):
        grid3D[winds["u"][0, :], winds["u"][1, :], i] = 3
        grid3D[winds["d"][0, :], winds["d"][1, :], i] = 3
        grid3D[winds["l"][0, :], winds["l"][1, :], i] = 3
        grid3D[winds["r"][0, :], winds["r"][1, :], i] = 3
        #debug_print(walls, winds, curr_pos)
        # Move winds
        winds["u"][1, :] -= 1
        winds["u"][winds["u"] == 0] = height - 2
        winds["d"][1, :] += 1
        winds["d"][winds["d"] == height - 1] = 1
        winds["l"][0, :] -= 1
        winds["l"][winds["l"] == 0] = width - 2
        winds["r"][0, :] += 1
        winds["r"][winds["r"] == width - 1] = 1

    grid3D[start[0], start[1], 0] = 2
    curr_state = 0
    counter = 0
    while True:
        counter += 1
        curr_state += 1
        curr_state %= wind_states

        i, j = np.nonzero(grid3D[..., curr_state - 1] == 2)
        grid3D[i, j, curr_state] |= 2

        grid3D[(i - 1).clip(0, width - 1), j, curr_state] |= 2
        grid3D[(i + 1).clip(0, width - 1), j, curr_state] |= 2
        grid3D[i, (j - 1).clip(0, height - 1), curr_state] |= 2
        grid3D[i, (j + 1).clip(0, height - 1), curr_state] |= 2

        if grid3D[goal[0], goal[1], curr_state] == 2:
            return counter


#example_answer = a(puzzle.example_data)
#print(example_answer)
example_answer = a(EXAMPLE_DATA)
print(example_answer)
assert example_answer == 18
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    exit()

example_answer = b(EXAMPLE_DATA)
print(example_answer)
assert example_answer == ...
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
