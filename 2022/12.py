import queue

import numpy as np
from aocd.models import Puzzle

YEAR = 2022
DAY = 12

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def parse_moutain_map(data):
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
    return grid, start, end, mountain_map


def a_star(start, end, grid, node_map):
    def h(node):
        return np.abs(end[0]- node[0]) + np.abs(end[1] - node[1])

    open_set = queue.PriorityQueue()
    open_set.put((0, start))

    g_score = np.full(grid.shape, np.inf)
    g_score[start] = 0

    f_score = np.full(grid.shape, np.inf)
    f_score[start] = h(start)

    while not open_set.empty():
        _, current = open_set.get()
        if (current[0] == end[0]) and (current[1] == end[1]):
            return int(g_score[current])

        for next_hill in node_map[current]:
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score[next_hill]:
                g_score[next_hill] = tentative_g_score
                f_score[next_hill] = tentative_g_score + h(next_hill)
                if next_hill not in open_set.queue:
                    open_set.put((f_score[next_hill], next_hill))


def a(data):
    grid, start, end, mountain_map = parse_moutain_map(data)
    return a_star(start, end, grid, mountain_map)

example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 31
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 437


# Part b
def b(data):
    grid, start, end, mountain_map = parse_moutain_map(data)
    scores = []
    for i, j in zip(*np.where(grid == 0)):
        scores.append(a_star((i, j), end, grid, mountain_map))
    return min([s for s in scores if s is not None])

example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 29
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 430
