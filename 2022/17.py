import datetime

import numpy as np
from aocd.models import Puzzle

np.set_printoptions(edgeitems=10, linewidth=180)

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)

EXAMPLE_DATA = ">>><<><>><<<>><>>><<<>>><<<><<<>><>><<>>"

SHAPES = [
    np.array(
        [
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1],
            [1],
            [1],
            [1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1],
            [1, 1],
        ],
        dtype=bool,
    ),
]

# Part a
def a(data, debug=False):
    grid_height = 4*2022+4 + 1
    grid_width = 7
    nbr_instructions = len(data)
    grid = np.zeros((grid_height, grid_width), dtype=int)
    grid[-1, :] = 2
    top = grid.shape[0] - 1
    move_nbr = 0
    for rock_nbr in range(2022 + 1):
        shape_nbr = rock_nbr % len(SHAPES)
        rock = SHAPES[shape_nbr]
        rock_height, rock_width = rock.shape
        i = top - rock_height - 3
        j = 2
        while True:
            curr_loc = np.s_[i:i + rock_height, j:j + rock_width]
            if debug:
                grid[curr_loc][rock] = 1
                print(grid[top - 4:, :])
                breakpoint()
                grid[curr_loc][rock] = 0
            # Move sideways
            move = 1 if data[move_nbr % nbr_instructions] == ">" else -1
            move_nbr += 1
            next_j = np.clip(j + move, 0, grid_width - rock_width)
            next_loc = np.s_[i:i + rock_height, next_j:next_j + rock_width]
            if (grid[next_loc][rock] == 2).any():
                # Hit something...
                next_j = j
            j = next_j
            curr_loc = np.s_[i:i + rock_height, j:j + rock_width]
            # Move down
            next_i = i + 1
            next_loc = np.s_[next_i:next_i + rock_height, j:j + rock_width]
            if (grid[next_loc][rock] == 2).any():
                # Hit something...
                grid[curr_loc][rock] = 2
                break
            i = next_i
        top = min(top, i)
        if debug:
            print(grid[top - 4:, :])
            breakpoint()
    return grid_height - top - 3


example_answer = a(EXAMPLE_DATA)
print(example_answer)
assert example_answer == 3068
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
