import re
from collections import defaultdict

import numpy as np
from aocd.models import Puzzle
from scipy.ndimage import watershed_ift
from scipy.signal import convolve2d

YEAR = 2023
DAY = 3

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    s = 0
    grid = np.vstack(
        [np.frombuffer(n.encode(), dtype=np.uint8) - ord("0") for n in data.split("\n")]
    )
    grid = np.pad(grid, 1, constant_values=254)  # FIXME: Why is this needed..?
    symbols = np.logical_and(grid != 254, grid > 9)
    kernel = np.ones((3, 3), np.uint8)
    hack = (
        convolve2d(
            symbols.astype(np.uint8),
            kernel,
            mode="same",
            boundary="fill",
        )
        > 0
    )
    tmp = np.ones(grid.shape, dtype=np.uint8) * 2
    digits = grid < 10
    tmp[digits] = 0
    good_digits = np.logical_and(digits, hack)
    tmp[good_digits] = 1
    markers = good_digits.astype(int)
    background = grid > 9
    markers[background] = -1
    ww = watershed_ift(
        tmp, markers
    )  # , structure=np.array([[0, 0, 0],[1, 1, 1],[0, 0, 0]]))
    good_digits = ww > 0
    grid = grid.astype(int)
    grid[~good_digits] = -1
    for row in grid:
        tmp = "".join(str(c) for c in row)
        s += sum([int(e) for e in tmp.split("-1") if e])
    return s


def a_alt(data):
    grid = np.vstack(
        [np.frombuffer(n.encode(), dtype=np.uint8) - ord("0") for n in data.split("\n")]
    )
    height, width = grid.shape
    grid = np.pad(grid, 1, constant_values=254)
    s_bad = 0
    for y in range(1, height + 1):
        bad_numbers = []
        candidate = []
        looking = False
        for x in range(0, width + 1):
            if (
                (grid[y, x] == 254)
                and (grid[y - 1, x] == 254)
                and (grid[y + 1, x] == 254)
            ):
                if candidate:
                    s_bad += int("".join(str(n) for n in candidate))
                looking = True
                candidate = []
            elif (
                looking
                and (grid[y, x] < 10)
                and (grid[y - 1, x] == 254)
                and (grid[y + 1, x] == 254)
            ):
                candidate.append(grid[y, x])
            else:
                candidate = []
                looking = False
        if looking and candidate:
            s_bad += int("".join(str(n) for n in candidate))
    s_total = sum(int(n) for n in re.findall(r"(\d+)", data))
    return s_total - s_bad


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 546312


# Part b
def b(data):
    lines = data.splitlines()
    lines = [f".{line}." for line in lines]
    width = len(lines[0])
    lines.insert(0, "." * width)
    lines.append("." * width)
    height = len(lines)
    maybe_gears = defaultdict(list)
    for y in range(1, height):
        number = []
        for x in range(1, width):
            if "0" <= lines[y][x] <= "9":
                number.append(lines[y][x])
            else:
                if number:
                    for j in range(y - 1, y + 2):
                        for i in range(x - len(number) - 1, x + 1):
                            if lines[j][i] == "*":
                                n = int("".join(number))
                                maybe_gears[(j, i)].append(n)
                number = []
    tmp = set(tuple(sorted(e)) for e in maybe_gears.values() if len(e) == 2)
    s = 0
    for t in tmp:
        s += t[0] * t[1]
    return s


example = """467..114..
...*......
..35..633.
......#...
617*......
.....+.58.
..592.....
......755.
...$.*....
.664.598.."""

assert b(example) == 467835

answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 87449461
