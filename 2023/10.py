import numpy as np
from aocd.models import Puzzle
from scipy import ndimage

YEAR = 2023
DAY = 10

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    grid = np.vstack(
        [np.frombuffer(row.encode(), dtype=np.uint8) for row in data.splitlines()]
    )
    grid = np.pad(grid, 1, constant_values=ord("."))
    start_pos = np.where(grid == ord("S"))
    start_pos = (start_pos[0][0], start_pos[1][0])
    s = 0
    pos = start_pos
    prev_pos = (-1, -1)
    prev_pipe = ord("S")
    while True:
        s += 1
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
            print(grid[i - 1 : i + 2, j - 1 : j + 2])
            breakpoint()
        prev_pos = pos
        prev_pipe = grid[prev_pos]
        pos = (i, j)
        if pos == start_pos:
            break
    return s // 2


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
    start_pos = np.where(grid == ord("S"))
    start_pos = (start_pos[0][0], start_pos[1][0])
    pos = start_pos
    prev_pos = (-1, -1)
    prev_pipe = ord("S")
    path_map = grid.copy()
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
            print(grid[i - 1 : i + 2, j - 1 : j + 2])
            breakpoint()
        prev_pos = pos
        path_map[pos] = ord("X")
        prev_pipe = grid[prev_pos]
        pos = (i, j)
        if pos == start_pos:
            break
    tmp = np.zeros(grid.shape)
    tmp[path_map == ord("X")] = grid[path_map == ord("X")]
    tmp2 = np.zeros(np.array(tmp.shape) * 2)
    tmp2[::2, ::2] = tmp
    tmp3 = tmp2.copy()
    for i in range(1, tmp2.shape[0] - 1):
        for j in range(1, tmp2.shape[1] - 1):
            if tmp2[i, j] == 0:
                if (tmp2[i, j + 1] in [ord("-"), ord("J"), ord("7")]) or (
                    tmp2[i, j - 1] in [ord("-"), ord("L"), ord("F")]
                ):
                    tmp3[i, j] = ord("-")
                elif (tmp2[i - 1, j] in [ord("|"), ord("7"), ord("F")]) or (
                    tmp2[i + 1, j] in [ord("|"), ord("L"), ord("J")]
                ):
                    tmp3[i, j] = ord("|")
            else:
                tmp3[i, j] = tmp2[i, j]

    tmp35 = ndimage.binary_fill_holes(tmp3)
    tmp36 = tmp35[::2, ::2]
    return tmp36.sum() - (tmp > 0).sum()


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
