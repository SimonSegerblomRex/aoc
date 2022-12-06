import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import convolve2d
from aocd.models import Puzzle

YEAR = 2021
DAY = 11

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    grid = np.frombuffer(data.replace("\n", "").encode(), dtype=np.uint8).reshape(
        10, 10
    ) - ord("0")
    flashes = 0
    for _ in range(100):
        grid += 1
        while True:
            flash = grid > 9
            if not flash.any():
                break
            grid[flash] = 0
            if 1:
                hack = (
                    uniform_filter(
                        flash.astype(float),
                        size=3,
                        mode="constant",
                        cval=0,
                    )
                    * 9 + 0.5
                ).astype(np.uint8)
            else:
                hack = convolve2d(
                    flash.astype(np.uint8),
                    np.ones((3, 3), dtype=np.uint8),
                    mode="same",
                    boundary="fill",
                )
            grid[grid > 0] += hack[grid > 0]
        flashes += (grid == 0).sum()
    return flashes


example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 1656
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 1700

# Part b
def b(data):
    grid = np.frombuffer(data.replace("\n", "").encode(), dtype=np.uint8).reshape(
        10, 10
    ) - ord("0")
    step = 0
    while True:
        step += 1
        grid += 1
        while True:
            flash = grid > 9
            if not flash.any():
                break
            grid[flash] = 0
            if 1:
                hack = (
                    uniform_filter(
                        flash.astype(float),
                        size=3,
                        mode="constant",
                        cval=0,
                    )
                    * 9 + 0.5
                ).astype(np.uint8)
            else:
                hack = convolve2d(
                    flash.astype(np.uint8),
                    np.ones((3, 3), dtype=np.uint8),
                    mode="same",
                    boundary="fill",
                )
            grid[grid > 0] += hack[grid > 0]
        if (grid == 0).all():
            return step


example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 195
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 273
