import numpy as np
from aocd.models import Puzzle

YEAR = 2023
DAY = 13

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def find_reflection_cols(grid):
    cols = np.flatnonzero((np.diff(grid, axis=1) == 0).all(axis=0))
    if len(cols) == 0:
        return [0]
    print(cols)
    out = []
    for col in cols:
        height, width = grid.shape
        cols_to_check = min(col + 1, width - 1 - col)
        if col + 1 < cols_to_check:
            if (
                np.abs(
                    grid[:, :cols_to_check]
                    - grid[:, col + 1 : col + 1 + cols_to_check][:, ::-1]
                ).sum()
                == 0
            ).all():
                out.append(col + 1)
        else:
            if (
                np.abs(
                    grid[:, col + 1 - cols_to_check : col + 1]
                    - grid[:, col + 1 : col + 1 + cols_to_check][:, ::-1]
                ).sum()
                == 0
            ).all():
                out.append(col + 1)
    if len(out) > 0:
        return out
    return [0]


def a(data):
    total = 0
    for subdata in data.split("\n\n"):
        grid = np.vstack(
            [np.frombuffer(row.encode(), dtype=np.int8) for row in subdata.splitlines()]
        )
        grid[grid == ord(".")] = 0
        grid[grid == ord("#")] = 1
        height, width = grid.shape
        total += sum(find_reflection_cols(grid)) + 100 * sum(
            find_reflection_cols(grid.T)
        )
    return total


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 33728


# Part b
def find_smudge_reflection_cols(grid):
    cols = np.flatnonzero(np.abs(np.diff(grid, axis=1)).sum(axis=0) <= 1)
    if len(cols) == 0:
        return [0]
    print(cols)
    out = []
    for col in cols:
        height, width = grid.shape
        cols_to_check = min(col + 1, width - 1 - col)
        if col + 1 < cols_to_check:
            if (
                np.abs(
                    grid[:, :cols_to_check]
                    - grid[:, col + 1 : col + 1 + cols_to_check][:, ::-1]
                ).sum()
                == 1
            ).all():
                out.append(col + 1)
        else:
            if (
                np.abs(
                    grid[:, col + 1 - cols_to_check : col + 1]
                    - grid[:, col + 1 : col + 1 + cols_to_check][:, ::-1]
                ).sum()
                == 1
            ).all():
                out.append(col + 1)
    if len(out) > 0:
        return out
    return [0]


def b(data):
    total = 0
    for subdata in data.split("\n\n"):
        grid = np.vstack(
            [np.frombuffer(row.encode(), dtype=np.int8) for row in subdata.splitlines()]
        )
        grid[grid == ord(".")] = 0
        grid[grid == ord("#")] = 1
        height, width = grid.shape
        total += sum(find_smudge_reflection_cols(grid)) + 100 * sum(
            find_smudge_reflection_cols(grid.T)
        )
    return total


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
