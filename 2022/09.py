import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)

EXAMPLE = """R 4
U 4
L 3
D 1
R 4
D 1
L 5
R 2
"""

MOVES = {
    "R": (1, 0),
    "U": (0, 1),
    "L": (-1, 0),
    "D": (0, -1),
}

def get_new_tail_coord(h, t):
    diff = h - t
    if (np.abs(diff[0]) <=1) and (np.abs(diff[1]) <= 1):
        return t
    if np.abs(diff[0]) == 2:
        return h - (np.sign(diff[0]), 0)
    if np.abs(diff[1]) == 2:
        return h - (0, np.sign(diff[1]))

# Part a
def a(data):
    data = data.splitlines()
    data = (line.split(" ") for line in data)
    moves = [(c, int(s)) for c, s in data]
    coords_head = [np.array((0, 0), dtype=int)]
    coords_tail1 = [np.array((0, 0), dtype=int)]
    coords_tail2 = [np.array((0, 0), dtype=int)]
    coords_tail3 = [np.array((0, 0), dtype=int)]
    coords_tail4 = [np.array((0, 0), dtype=int)]
    coords_tail5 = [np.array((0, 0), dtype=int)]
    coords_tail6 = [np.array((0, 0), dtype=int)]
    coords_tail7 = [np.array((0, 0), dtype=int)]
    coords_tail8 = [np.array((0, 0), dtype=int)]
    coords_tail9 = [np.array((0, 0), dtype=int)]
    for move in moves:
        for step in range(move[1]):
            coords_head.append(coords_head[-1] + MOVES[move[0]])
            coords_tail1.append(get_new_tail_coord(coords_head[-1], coords_tail1[-1]))
            coords_tail2.append(get_new_tail_coord(coords_tail1[-1], coords_tail2[-1]))
            coords_tail3.append(get_new_tail_coord(coords_tail2[-1], coords_tail3[-1]))
            coords_tail4.append(get_new_tail_coord(coords_tail3[-1], coords_tail4[-1]))
            coords_tail5.append(get_new_tail_coord(coords_tail4[-1], coords_tail5[-1]))
            coords_tail6.append(get_new_tail_coord(coords_tail5[-1], coords_tail6[-1]))
            coords_tail7.append(get_new_tail_coord(coords_tail6[-1], coords_tail7[-1]))
            coords_tail8.append(get_new_tail_coord(coords_tail7[-1], coords_tail8[-1]))
            coords_tail9.append(get_new_tail_coord(coords_tail8[-1], coords_tail9[-1]))
    if 0:
        tmp = np.zeros((6, 6), dtype=int)
        for coord in coords_tail9:
            tmp[coord[1], coord[0]] = 1
        print(np.flipud(tmp))
    with open("out.txt", "w") as f:
        for h, t in zip(coords_head, coords_tail9):
            f.write(f"h: {h}, t: {t}\n")
    #breakpoint()
    return len(set(map(tuple, coords_tail9)))

example_answer = a(EXAMPLE)
print(example_answer)
assert example_answer == 1
answer = a(puzzle.input_data)
print("a:", answer)
#puzzle.answer_a = answer


# Part b
def b(data):
    exit()

example_answer = b(EXAMPLE)
print(example_answer)
assert example_answer == ...
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
