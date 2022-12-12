import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    grid = np.vstack([np.frombuffer(line.encode(), dtype=np.uint8) for line in data.splitlines()]) - ord("a")
    grid = grid.astype(int)
    start = np.where(grid == 242)
    end = np.where(grid == 228)
    grid[start] = 0
    grid[end] = 25
    height, width = grid.shape
    mountain_map = {}
    def check_height(c0, c1):
        #return (grid[c1] - grid[c0]) < 2
        return ((grid[c1] - grid[c0]) < 2) and ((grid[c1] - grid[c0]) > -1) #FIXME: Är det andra antagendet ok..?
    for y, x in np.ndindex(grid.shape):
        tmp = []
        if (y > 0):
            tmp.append((y - 1, x))
        if x > 0:
            tmp.append((y, x - 1))
        if y < height - 1:
            tmp.append((y + 1, x))
        if x < width - 1:
            tmp.append((y, x + 1))
        tmp = [c for c in tmp if check_height((y, x), c)]
        mountain_map[(y, x)] = tmp

    paths = []
    best_score = [np.inf]
    def find_paths(path):
        curr_hill = path[-1]
        for next_hill in mountain_map[curr_hill]:
            #if (next_hill[0] == start[0]) and (next_hill[1] == start[1]):
            #    continue
            if next_hill in path:
                continue
            new_path = [*path, next_hill]
            """
            if next_hill in small_hills:
                small_visit_counts = [new_path.count(c) for c in small_hills]
                if (small_visit_counts.count(2) > 1) or max(small_visit_counts) > 2:
                    continue
            """
            curr_score = len(new_path)
            if curr_score > best_score[0]:
                continue
            if curr_score + np.abs(end[0] - next_hill[0]) + np.abs(end[1] - next_hill[1]) > best_score:
                continue
            if (next_hill[0] == end[0]) and (next_hill[1] == end[1]):
                best_score[0] = curr_score
                paths.append(new_path)
                return  # FIXME: OK..?
            else:
                find_paths(new_path)
        return paths

    find_paths([(0, 0)])
    return best_score[0] -1

example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 31
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    exit()

example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == ...
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
