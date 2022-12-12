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
    start = tuple(np.array(np.where(grid == 242)).flatten())
    end = tuple(np.array(np.where(grid == 228)).flatten())
    grid[start] = 0
    grid[end] = 25
    height, width = grid.shape
    mountain_map = {}
    def check_height(c0, c1):
        return (grid[c1] - grid[c0]) < 2
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
    dead_end = []
    def find_paths(path):
        curr_hill = path[-1]
        for next_hill in mountain_map[curr_hill]:
            #if (next_hill[0] == start[0]) and (next_hill[1] == start[1]):
            #    continue
            if next_hill in path:
                continue
            if next_hill in dead_end:
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
                print([len(p) for p in paths])
            else:
                find_paths(new_path)
            #dead_end.append(next_hill)

    if 0:
        find_paths([(0, 0)])
    else:
        def a_star(start, end):
            def h(node):
                return np.abs(end[0]- node[0]) + np.abs(end[1] - node[1])

            open_set = [start]

            path = [start]

            g_score = np.full(grid.shape, np.inf)
            g_score[start] = 0

            f_score = np.full(grid.shape, np.inf)
            f_score[start] = h(start)

            while open_set:
                current = open_set.pop(0)
                if (current[0] == end[0]) and (current[1] == end[1]):
                    return int(g_score[current])

                for next_hill in mountain_map[current]:
                    tentative_g_score = g_score[current] + 1
                    if tentative_g_score < g_score[next_hill]:
                        #came_from[]  # Behövs inte?
                        g_score[next_hill] = tentative_g_score
                        f_score[next_hill] = tentative_g_score + h(next_hill)
                        if next_hill not in open_set:
                            open_set.append(next_hill)

                    #breakpoint()
        tmp = a_star(start, end)
    return tmp

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
