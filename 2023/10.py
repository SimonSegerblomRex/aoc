import numpy as np
from aocd.models import Puzzle
from scipy import ndimage

YEAR = 2023
DAY = 10

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def get_loop_mask(grid):
    start_pos = np.where(grid == ord("S"))
    start_pos = (start_pos[0][0], start_pos[1][0])
    pos = start_pos
    prev_pos = (-1, -1)
    loop = np.full(grid.shape, False, dtype=bool)
    while True:
        i, j = pos
        if (
            grid[i, j + 1] in [ord("-"), ord("J"), ord("7"), ord("S")]
            and prev_pos != (i, j + 1)
            and grid[i, j] in [ord("-"), ord("L"), ord("F"), ord("S")]
        ):
            j = j + 1
        elif (
            grid[i - 1, j] in [ord("|"), ord("7"), ord("F"), ord("S")]
            and prev_pos != (i - 1, j)
            and grid[i, j] in [ord("|"), ord("L"), ord("J"), ord("S")]
        ):
            i = i - 1
        elif (
            grid[i, j - 1] in [ord("-"), ord("L"), ord("F"), ord("S")]
            and prev_pos != (i, j - 1)
            and grid[i, j] in [ord("-"), ord("J"), ord("7"), ord("S")]
        ):
            j = j - 1
        elif (
            grid[i + 1, j] in [ord("|"), ord("L"), ord("J"), ord("S")]
            and prev_pos != (i + 1, j)
            and grid[i, j] in [ord("|"), ord("F"), ord("7"), ord("S")]
        ):
            i = i + 1
        else:
            raise RuntimeError("Shouldn't end up here...")
        prev_pos = pos
        loop[pos] = ord("X")
        pos = (i, j)
        if pos == start_pos:
            break
    return loop


def a(data):
    grid = np.vstack(
        [np.frombuffer(row.encode(), dtype=np.uint8) for row in data.splitlines()]
    )
    grid = np.pad(grid, 1, constant_values=ord("."))
    loop_mask = get_loop_mask(grid)
    return np.sum(loop_mask) // 2


example = """.....
.S-7.
.|.|.
.L-J.
....."""
example_answer = a(example)
assert example_answer == 4


example = """..F7.
.FJ|.
SJ.L7
|F--J
LJ..."""
example_answer = a(example)
assert example_answer == 8

answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 6979


# Part b
def b(data):
    grid = np.vstack(
        [np.frombuffer(row.encode(), dtype=np.uint8) for row in data.splitlines()]
    )
    grid = np.pad(grid, 1, constant_values=ord("."))
    loop_mask = get_loop_mask(grid)
    loop = np.zeros(grid.shape, dtype=int)
    loop[loop_mask] = grid[loop_mask]
    upscaled_loop = np.zeros(np.array(loop.shape) * 2)
    upscaled_loop[::2, ::2] = loop
    for i in range(1, upscaled_loop.shape[0] - 1):
        for j in range(1, upscaled_loop.shape[1] - 1):
            if upscaled_loop[i, j] == 0:
                if (upscaled_loop[i, j + 1] in [ord("-"), ord("J"), ord("7")]) or (
                    upscaled_loop[i, j - 1] in [ord("-"), ord("L"), ord("F")]
                ):
                    upscaled_loop[i, j] = ord("-")
                elif (upscaled_loop[i - 1, j] in [ord("|"), ord("7"), ord("F")]) or (
                    upscaled_loop[i + 1, j] in [ord("|"), ord("L"), ord("J")]
                ):
                    upscaled_loop[i, j] = ord("|")
    filled_upscaled_loop = ndimage.binary_fill_holes(upscaled_loop)
    filled_loop = filled_upscaled_loop[::2, ::2]
    if 0:
        import matplotlib.pyplot as plt

        plt.imshow(filled_loop)
        plt.figure()
        plt.imshow(filled_upscaled_loop)
        plt.figure()
        plt.imshow(upscaled_loop)
        plt.figure()
        plt.imshow(loop)
        plt.show()
    return filled_loop.sum() - (loop > 0).sum()


example = """...........
.S-------7.
.|F-----7|.
.||.....||.
.||.....||.
.|L-7.F-J|.
.|..|.|..|.
.L--J.L--J.
..........."""
example_answer = b(example)
assert example_answer == 4


example = """.F----7F7F7F7F-7....
.|F--7||||||||FJ....
.||.FJ||||||||L7....
FJL7L7LJLJ||LJ.L-7..
L--J.L7...LJS7F-7L7.
....F-J..F7FJ|L7L7L7
....L7.F7||L7|.L7L7|
.....|FJLJ|FJ|F7|.LJ
....FJL-7.||.||||...
....L---J.LJ.LJLJ..."""
example_answer = b(example)
assert example_answer == 8


example = """FF7FSF7F7F7F7F7F---7
L|LJ||||||||||||F--J
FL-7LJLJ||||||LJL-77
F--JF--7||LJLJ7F7FJ-
L---JF-JLJ.||-FJLJJ7
|F|F-JF---7F7-L7L|7|
|FFJF7L7F-JF7|JL---7
7-L-JL7||F7|L7F-7F7|
L.L7LFJ|||||FJL7||LJ
L7JLJL-JLJLJL--JLJ.L"""
example_answer = b(example)
assert example_answer == 10

answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 443
