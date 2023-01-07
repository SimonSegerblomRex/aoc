import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 15

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def djikstra(start, end, grid):
    dist = np.full(grid.shape, np.inf)
    dist[start] = 0

    height, width = grid.shape

    def neighbours(node):
        if node[0] > 0:
            yield (node[0] - 1, node[1])
        if node[0] < height - 1:
            yield (node[0] + 1, node[1])
        if node[1] > 0:
            yield (node[0], node[1] - 1)
        if node[1] < width - 1:
            yield (node[0], node[1] + 1)

    to_visit = [start]
    while to_visit:
        curr_node = to_visit.pop(0)
        for node in neighbours(curr_node):
            tentative_dist = dist[curr_node] + grid[node]
            if tentative_dist < dist[node]:
                dist[node] = tentative_dist
                if node not in to_visit:
                    to_visit.append(node)

    return dist[end]


def a(data):
    grid = np.vstack([np.fromiter(row, dtype=int) for row in data.splitlines()])
    start = (0, 0)
    end = tuple(np.array(grid.shape) - 1)
    return int(djikstra(start, end, grid))


example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 40
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    grid = np.vstack([np.fromiter(row, dtype=int) for row in data.splitlines()])
    grid -= 1
    blocks = [grid + i + j for i, j in np.ndindex((5, 5))]
    blocks = [blocks[i:i + 5] for i in range(0, len(blocks), 5)]
    grid = np.block(blocks)
    grid %= 9
    grid += 1
    start = (0, 0)
    end = tuple(np.array(grid.shape) - 1)
    return int(djikstra(start, end, grid))


example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 315
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
