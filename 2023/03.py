import datetime
import re

import numpy as np
from aocd.models import Puzzle
from scipy.signal import convolve2d
from scipy.ndimage import watershed_ift

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a_old_broken(data):
    s_good = 0
    s_bad = 0
    hmm = []
    grid = np.vstack(
        [np.frombuffer(n.encode(), dtype=np.uint8) - ord("0") for n in data.split("\n")]
    )
    symbols = np.logical_and(grid != 254, grid > 9)
    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ])
    hack = convolve2d(
        symbols.astype(np.uint8),
        kernel,
        mode="same",
        boundary="fill",
    ) > 0
    digits = grid < 10
    good_digits = np.logical_and(digits, hack)
    background = grid > 9
    tmp = np.ones(grid.shape, dtype=np.uint8) * 3
    tmp[digits] = 1
    tmp[good_digits] = 2
    markers = good_digits.astype(int) * 2
    markers[background] = -1
    ww = watershed_ift(tmp, markers, structure=np.array([[0,0,0],[1, 1, 1],[0,0,0]]))
    good_digits = ww > 0
    for row, good in zip(grid.copy(), good_digits):
        tt = row.copy().astype(int)
        tt[~good] = -1
        tmp = "".join(str(c) for c in tt)
        print([int(e) for e in tmp.split("-1") if e])
        hmm.extend([int(e) for e in tmp.split("-1") if e])
        s_good += sum([int(e) for e in tmp.split("-1") if e])
    bad_digits = np.logical_and(digits, ~good_digits)
    for row, good in zip(grid.copy(), bad_digits):
        tt = row.copy().astype(int)
        tt[~good] = -1
        tmp = "".join(str(c) for c in tt)
        print([int(e) for e in tmp.split("-1") if e])
        s_bad += sum([int(e) for e in tmp.split("-1") if e])
    s_total = sum(int(n) for n in re.findall(r"(\d+)", data))
    print(s_total, s_good, s_bad)
    assert s_total == s_good + s_bad
    for iy, ix in np.ndindex(grid.shape):
        if iy == 0:
            continue
        try:
            if (0<= grid[iy, ix] <= 9) and ((0<= grid[iy- 1, ix] <= 9) or (0<= grid[iy + 1, ix] <= 9)):
                breakpoint()
        except IndexError:
            continue
    #assert len(np.unique(hmm)) == len(hmm)
    #return sum(np.unique(hmm))
    #breakpoint()
    return s_good

def a(data):
    grid = np.vstack(
        [np.frombuffer(n.encode(), dtype=np.uint8) - ord("0") for n in data.split("\n")]
    )
    height, width = grid.shape
    grid = np.pad(grid, 1, constant_values = 254)
    s_bad = 0
    for y in range(1, height + 1):
        bad_numbers = []
        candidate = []
        looking = False
        for x in range(0, width + 1):
            if (grid[y, x] == 254) and (grid[y - 1, x] == 254) and (grid[y + 1, x] == 254):
                if candidate:
                    s_bad += int("".join(str(n) for n in candidate))
                looking = True
                candidate = []
            elif looking and (grid[y, x] < 10) and (grid[y - 1, x] == 254) and (grid[y + 1, x] == 254):
                candidate.append(grid[y, x])
            else:
                candidate = []
                looking = False
        if looking and candidate:
            s_bad += int("".join(str(n) for n in candidate))
    s_total = sum(int(n) for n in re.findall(r"(\d+)", data))
    return s_total - s_bad
    breakpoint()

for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a

if 0:
    example2 = """407..114..
    ...*......
    ..35..633.
    .....1#..*
    617*.....2
    .....+.58.
    ..592...22
    ..9...755-
    ...$2*....
    .664.598.."""
    print(a(example2), 4361 - 60 + 9 + 1 + 2 + 2 + 22, "AAAAAAAAAAA")
    assert a(example2) == 4361 - 60 + 9 + 1 + 2 + 2 + 22
    breakpoint()

answer = a(puzzle.input_data)
print("a:", answer)
print("old:", a_old_broken(puzzle.input_data), " vs ", answer)
assert answer > 457353
puzzle.answer_a = answer

breakpoint()
# Part b
def b(data):
    print(data)
    breakpoint()


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
