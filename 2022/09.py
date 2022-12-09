import numpy as np
from aocd.models import Puzzle

YEAR = 2022
DAY = 9

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

EXAMPLE2 = """R 5
U 8
L 8
D 3
R 17
D 10
L 25
U 20
"""

MOVES = {
    "R": (1, 0),
    "U": (0, 1),
    "L": (-1, 0),
    "D": (0, -1),
}


def get_new_tail_coord(h, t):
    diff = h - t
    if (np.abs(diff[0]) <= 1) and (np.abs(diff[1]) <= 1):
        return t
    if (np.abs(diff[0]) == 2) and (np.abs(diff[1]) == 2):
        return h - (np.sign(diff[0]), np.sign(diff[1]))
    if np.abs(diff[0]) == 2:
        return h - (np.sign(diff[0]), 0)
    if np.abs(diff[1]) == 2:
        return h - (0, np.sign(diff[1]))
    print("Shouldn't end up here...")
    breakpoint()


# Part a
def a(data, tail_length, debug=False):
    data = data.splitlines()
    data = (line.split(" ") for line in data)
    moves = [(c, int(s)) for c, s in data]
    coords_head = [np.array((0, 0), dtype=int)]
    coords_tail = [[np.array((0, 0), dtype=int)] for _ in range(tail_length)]
    for move in moves:
        for _ in range(move[1]):
            coords_head.append(coords_head[-1] + MOVES[move[0]])
            coords_tail[0].append(
                get_new_tail_coord(coords_head[-1], coords_tail[0][-1])
            )
            for i in range(1, tail_length):
                coords_tail[i].append(
                    get_new_tail_coord(coords_tail[i - 1][-1], coords_tail[i][-1])
                )
            if debug:
                tmp = np.zeros((6, 6), dtype=int)
                tmp[coords_head[-1][1], coords_head[-1][0]] = 10
                for i in range(tail_length):
                    tmp[coords_tail[i][-1][1], coords_tail[i][-1][0]] = i + 1
                print(np.flipud(tmp))
                breakpoint()
    return len(set(map(tuple, coords_tail[-1])))


example_answer = a(EXAMPLE, tail_length=1)
print(example_answer)
assert example_answer == 13
answer = a(puzzle.input_data, tail_length=1)
print("a:", answer)
assert answer == 5902


# Part b
example_answer = a(EXAMPLE, tail_length=9)  # , debug=True)
print(example_answer)
assert example_answer == 1
example_answer = a(EXAMPLE2, tail_length=9)
print(example_answer)
assert example_answer == 36
answer = a(puzzle.input_data, tail_length=9)
print("b:", answer)
assert answer == 2445
