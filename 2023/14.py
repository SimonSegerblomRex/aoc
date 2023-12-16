import re

from aocd.models import Puzzle

YEAR = 2023
DAY = 14

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def tilt_row(row):
    row = "".join(row)
    matches = re.finditer(r"([O.]+)", row)
    out = list(row.replace("O", "."))
    for m in matches:
        n = m.group(1).count("O")
        if n:
            out[m.start() : m.start() + n] = "O" * n
    return out


def rotate_grid(grid, k):
    if k == -1:
        return list(zip(*reversed(grid)))
    if k == 1:
        return list(zip(*grid))[::-1]
    raise ValueError


def tilt_grid(grid):
    grid = rotate_grid(grid, 1)
    new_grid = []
    for row in grid:
        new_row = tilt_row(row)
        new_grid.append(new_row)
    return rotate_grid(new_grid, -1)


def score_grid(grid):
    scores = range(len(grid), 0, -1)
    return sum(r.count("O") * s for r, s in zip(grid, scores))


def a(data):
    grid = tilt_grid([list(line) for line in data.splitlines()])
    return score_grid(grid)


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 108759


# Part b
def b(data, rotations, cache=True):
    grid_cache = {}
    curr_grid = [list(line) for line in data.splitlines()]
    for i in range(rotations):
        curr_grid = tilt_grid(curr_grid)
        if cache and not (i + 1) % 4:
            grid_hash = tuple(curr_grid)
            if grid_hash in grid_cache:
                to_go = (rotations - i) % (i - grid_cache[grid_hash])
                return b("\n".join("".join(row) for row in curr_grid), to_go, cache=False)
            grid_cache[grid_hash] = i
        curr_grid = rotate_grid(curr_grid, -1)
    return score_grid(curr_grid)


example_answer = b(example.input_data, 1000000000 * 4)
print(f"Example answer: {example_answer} (expecting: {64})")
assert example_answer == 64
answer = b(puzzle.input_data, 1000000000 * 4)
print("b:", answer)
assert answer == 89089
