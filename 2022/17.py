import numpy as np
from aocd.models import Puzzle

np.set_printoptions(edgeitems=10, linewidth=180)

YEAR = 2022
DAY = 17

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
def a(data, nbr_rocks, debug=False):
    grid_height = 10000
    grid_width = 7
    nbr_instructions = len(data)
    grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
    grid[-1, :] = 2
    top = grid.shape[0] - 1
    move_nbr = 0
    reduced_height = 0
    for rock_nbr in range(nbr_rocks + 1):
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
            # Check if we can reduce size...
            tmp = np.argmax(grid, axis=0).max()
            if tmp < grid.shape[0] - 1000:
                height_reduction = grid.shape[0] - tmp - 1
                reduced_height += height_reduction
                grid = np.roll(grid, height_reduction, axis=0)
                grid[:height_reduction, :] = 0
                i += height_reduction
                top += height_reduction
        top = min(top, i)

        if debug:
            print(grid[top - 4:, :])
            breakpoint()
    return grid_height - top - 3 + reduced_height


example_answer = a(EXAMPLE_DATA, nbr_rocks=2022)
print(example_answer)
assert example_answer == 3068
answer = a(puzzle.input_data, nbr_rocks=2022)
print("a:", answer)
assert answer == 3119

# Part b
example_answer = a(EXAMPLE_DATA, nbr_rocks=1000000000000)
print(example_answer)
assert example_answer == 1514285714288
answer = a(puzzle.input_data, nbr_rocks=1000000000000)
print("b:", answer)
puzzle.answer_b = answer
