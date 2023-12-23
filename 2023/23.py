import datetime
import re
import queue
from collections import defaultdict
import sys
from functools import cache

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


sys.setrecursionlimit(100000)


# Part a
def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from.pop(current)
        total_path.append(current)
    return total_path


def debug_print(forest, path):
    height, width = int(forest[-1].real + 1), int(forest[-1].imag + 1)
    for i in range(height):
        for j in range(width):
            if complex(i, j) in forest:
                print("#", end="")
            elif complex(i, j) in path:
                print("O", end="")
            else:
                print(".", end="")
        print("")


def b_star(paths, end, forest, slopes):
    if sum(p[-1] == end for p in paths) == len(paths):
        return paths
    new_paths = []
    for path in paths:
        current = path[-1]
        if current == end:
            new_paths.append(path)
            continue
        dirs = [0 + 1j, -1 + 0j, 0 - 1j, 1 + 0j]
        for dir in dirs:
            next_node = current + dir
            if next_node in forest:
                continue
            if next_node in path:
                continue
            if next_node in slopes:
                if dir != slopes[next_node]:
                    continue
            new_paths.append([*path, next_node])
    return b_star(new_paths, end, forest, slopes)


def a(data):
    forest = []
    slopes = {}
    dirs = {
        "^": -1 + 0j,
        ">": 0 + 1j,
        "v": 1 + 0j,
        "<": 0 - 1j,
    }
    for i, line in enumerate(data.splitlines()):
        for m in re.finditer("#", line):
            j = m.start()
            forest.append(complex(i, j))
        for m in re.finditer("\^|>|v|<", line):
            j = m.start()
            slopes[complex(i, j)] = dirs[m.group(0)]
    height, width = i + 1, j + 1

    start = 0 + 1j
    dir = 1 + 0j

    goal = complex(height - 1, width - 2)

    forest.append(start - 1)
    forest.append(goal + 1)

    paths =  b_star([[start]], goal, forest, slopes)
    return max(len(p) for p in paths) - 1


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {94})")
        assert example_answer == 94
#answer = a(puzzle.input_data)
#print("a:", answer)
#puzzle.answer_a = answer


# Part b
def c_star(path, end, forest, slopes):
    if path[-1] == end:
        return path
    current = path[-1]
    candidates = []
    dirs = [0 + 1j, -1 + 0j, 0 - 1j, 1 + 0j]
    for dir in dirs:
        next_node = current + dir
        if next_node in forest:
            continue
        if next_node in path:
            continue
        candidate = c_star([*path, next_node], end, forest, slopes)
        candidates.append(candidate)
    best_path = []
    max_length = 0
    for p in candidates:
        if p and p[-1] == end:
            if len(p) > max_length:
                best_path = p
    return best_path


forest = []

@cache
def d_star(path, end):
    if path[-1] == end:
        return path
    current = path[-1]
    candidates = []
    dirs = [0 + 1j, -1 + 0j, 0 - 1j, 1 + 0j]
    for dir in dirs:
        next_node = current + dir
        if next_node in forest:
            continue
        if next_node in path:
            continue
        candidate = d_star((*path, next_node), end)
        candidates.append(candidate)
    best_path = []
    max_length = 0
    for p in candidates:
        if p and p[-1] == end:
            if len(p) > max_length:
                best_path = p
    return best_path

def b(data):
    grid = np.vstack(
        [np.frombuffer(row.encode(), dtype=np.uint8) for row in data.splitlines()]
    )
    height, width = grid.shape
    grid[grid != ord("#")] = 0
    grid[grid == ord("#")] = 3
    grid[0, 1] = 3
    grid[0, 1] = 3
    grid[height-1, width-2] = 3

    if 0:
        if 0:
            from scipy.ndimage import watershed_ift
            markers = grid.copy().astype(int)
            markers[markers == 2] = -1
            markers[1,2] = 1
            markers[height-2,width-2] = 2
            grid = watershed_ift(grid, markers)
        elif 1:
            from skimage.segmentation import watershed
            markers = grid.copy().astype(int)
            markers[markers == 2] = -1
            markers[1,2] = 1
            markers[height-2,width-2] = 2
            grid = watershed(grid, markers)

        for i in range(height):
            for j in range(width):
                if grid[i, j] == 3:
                    print("#", end="")
                elif grid[i, j] == 1:
                    print("O", end="")
                elif grid[i, j] == 2:
                    print("X", end="")
                else:
                    print(".", end="")
            print("")

        breakpoint()

    global forest
    forest = []
    slopes = {}
    dirs = {
        "^": -1 + 0j,
        ">": 0 + 1j,
        "v": 1 + 0j,
        "<": 0 - 1j,
    }
    for i, line in enumerate(data.splitlines()):
        for m in re.finditer("#", line):
            j = m.start()
            forest.append(complex(i, j))
        for m in re.finditer("\^|>|v|<", line):
            j = m.start()
            slopes[complex(i, j)] = dirs[m.group(0)]
    height, width = i + 1, j + 1

    start = 0 + 1j
    dir = 1 + 0j

    goal = complex(height - 1, width - 2)

    forest.append(start - 1)
    forest.append(goal + 1)

    path = d_star((start,), goal)
    return len(path) - 1



example_answer = b(example.input_data)
print(f"Example answer: {example_answer} (expecting: {154})")
assert example_answer == 154
answer = b(puzzle.input_data)
print("b:", answer)
assert answer > 2000
puzzle.answer_b = answer
