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
            if np.all(walls == [[x], [y]], axis=0).any():
                print("#", end="")
            elif np.all(winds["u"] == [[x], [y]], axis=0).any():
                print("^", end="")
            elif np.all(winds["d"] == [[x], [y]], axis=0).any():
                print("v", end="")
            elif np.all(winds["l"] == [[x], [y]], axis=0).any():
                print("<", end="")
            elif np.all(winds["r"] == [[x], [y]], axis=0).any():
                print(">", end="")
            elif (x, y) == curr_pos:
                print("E", end="")
            else:
                # FIXME
                print(".", end="")
        print("")
    breakpoint()


def create_3D_grid(data):
    rows = [np.frombuffer(row.encode(), dtype=np.uint8) for row in data.splitlines()]
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
        # debug_print(walls, winds, start)
        # Move winds
        winds["u"][1, :] -= 1
        winds["u"][1, winds["u"][1, :] == 0] = height - 2
        winds["d"][1, :] += 1
        winds["d"][1, winds["d"][1, :] == height - 1] = 1
        winds["l"][0, :] -= 1
        winds["l"][0, winds["l"][0, :] == 0] = width - 2
        winds["r"][0, :] += 1
        winds["r"][0, winds["r"][0, :] == width - 1] = 1

    return grid3D, start, goal


def shortest_path(grid3D_orig, start, goal, curr_state):
    grid3D = grid3D_orig.copy()
    grid3D[start[0], start[1], curr_state] = 2
    counter = 0
    height, width = grid3D.shape[:2]
    wind_states = np.lcm(height - 2, width - 2)
    while True:
        counter += 1
        curr_state += 1
        curr_state %= wind_states

        i, j = np.nonzero(grid3D[..., curr_state - 1] == 2)
        grid3D[i, j, curr_state] |= 2

        grid3D[(i - 1).clip(0, height - 1), j, curr_state] |= 2
        grid3D[(i + 1).clip(0, height - 1), j, curr_state] |= 2
        grid3D[i, (j - 1).clip(0, width - 1), curr_state] |= 2
        grid3D[i, (j + 1).clip(0, width - 1), curr_state] |= 2

        if grid3D[goal[0], goal[1], curr_state] == 2:
            return counter, curr_state


def a(data):
    grid3D, start, goal = create_3D_grid(data)
    return shortest_path(grid3D, start, goal, 0)[0]


example_answer = a(EXAMPLE_DATA)
print(example_answer)
assert example_answer == 18
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 373


# Part b
def b(data):
    grid3D, start, goal = create_3D_grid(data)
    first, curr_state = shortest_path(grid3D, start, goal, 0)
    second, curr_state = shortest_path(grid3D, goal, start, curr_state)
    third, _ = shortest_path(grid3D, start, goal, curr_state)
    return first + second + third


example_answer = b(EXAMPLE_DATA)
print(example_answer)
assert example_answer == 54
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 997
