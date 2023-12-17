import queue
from collections import defaultdict

from aocd.models import Puzzle

YEAR = 2023
DAY = 17

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def create_grid(rows):
    grid = {}
    for i, row in enumerate(rows):
        for j, c in enumerate(row):
            pos = complex(i, j)
            grid[pos] = int(c)
    return grid


def a_star(start, end, grid):
    def h(node):
        return int(abs(end.real - node.real) + abs(end.imag - node.imag))

    open_set = queue.PriorityQueue()
    open_set.put((0, 0, start, 0 + 1j, 1))
    open_set.put((0, 1, start, 1 + 0j, 1))

    g_score = defaultdict(lambda: 1 << 30)
    g_score[(start, 0 + 1j, 1)] = 0
    g_score[(start, 1 + 0j, 1)] = 0

    f_score = defaultdict(lambda: 1 << 30)
    f_score[(start, 0 + 1j, 1)] = h(start)
    f_score[(start, 1 + 0j, 1)] = h(start)

    i = 2
    while not open_set.empty():
        _, _, current, prev_dir, count = open_set.get()
        if current == end:
            return int(g_score[(current, prev_dir, count)])

        dirs = [0 + 1j, -1 + 0j, 0 - 1j, 1 + 0j]
        dirs.remove(-prev_dir)
        for dir in dirs:
            next_node = current + dir
            if next_node not in grid:
                continue
            if prev_dir == dir:
                next_count = count + 1
            else:
                next_count = 1
            if next_count > 3:
                continue
            tentative_g_score = g_score[(current, prev_dir, count)] + grid[next_node]
            if tentative_g_score < g_score[(next_node, dir, next_count)]:
                g_score[(next_node, dir, next_count)] = tentative_g_score
                f_score[(next_node, dir, next_count)] = tentative_g_score + h(next_node)
                open_set.put(
                    (
                        f_score[(next_node, dir, next_count)],
                        i,
                        next_node,
                        dir,
                        next_count,
                    )
                )
                i += 1


def a(data):
    rows = data.splitlines()
    grid = create_grid(rows)
    start = list(grid)[0]
    end = list(grid)[-1]
    return a_star(start, end, grid)


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 959


# Part b
def a_star_b(start, end, grid):
    def h(node):
        return int(abs(end.real - node.real) + abs(end.imag - node.imag))

    open_set = queue.PriorityQueue()
    open_set.put((0, 0, start, 0 + 1j, 1))
    open_set.put((0, 1, start, 1 + 0j, 1))

    g_score = defaultdict(lambda: 1 << 30)
    g_score[(start, 0 + 1j, 1)] = 0
    g_score[(start, 1 + 0j, 1)] = 0

    f_score = defaultdict(lambda: 1 << 30)
    f_score[(start, 0 + 1j, 1)] = h(start)
    f_score[(start, 1 + 0j, 1)] = h(start)

    i = 2
    while not open_set.empty():
        _, _, current, prev_dir, count = open_set.get()
        if current == end:
            return int(g_score[(current, prev_dir, count)])

        dirs = [0 + 1j, -1 + 0j, 0 - 1j, 1 + 0j]
        dirs.remove(-prev_dir)
        for dir in dirs:
            next_node = current + dir
            if next_node not in grid:
                continue
            if prev_dir == dir:
                next_count = count + 1
            else:
                if count >= 4:
                    next_count = 1
                else:
                    continue
            if next_count > 10:
                continue
            tentative_g_score = g_score[(current, prev_dir, count)] + grid[next_node]
            if tentative_g_score < g_score[(next_node, dir, next_count)]:
                g_score[(next_node, dir, next_count)] = tentative_g_score
                f_score[(next_node, dir, next_count)] = tentative_g_score + h(next_node)
                open_set.put(
                    (
                        f_score[(next_node, dir, next_count)],
                        i,
                        next_node,
                        dir,
                        next_count,
                    )
                )
                i += 1


def b(data):
    rows = data.splitlines()
    grid = create_grid(rows)
    start = list(grid)[0]
    end = list(grid)[-1]
    return a_star_b(start, end, grid)


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {94})")
        assert example_answer == 94
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 1135
