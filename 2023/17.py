from ast import Attribute
import queue
from collections import defaultdict
from telnetlib import TM

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


def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]


def a_star(start, end, grid, hist=3):
    def h(node):
        return int(abs(end.real - node.real) + abs(end.imag - node.imag))

    open_set = queue.PriorityQueue()
    open_set.put((0, 0, start, (None,)*hist))

    came_from = {}
    came_from[(start, (None,)*hist)] = None

    g_score = defaultdict(lambda: 1 << 30)
    g_score[(start, (None,)*hist)] = 0

    f_score = defaultdict(lambda: 1 << 30)
    f_score[(start, (None,)*hist)] = h(start)

    i = 1
    while not open_set.empty():
        _, _, current, prev_nodes = open_set.get()
        if current == end:
            #print(int(g_score[(current, prev_nodes)]))
            #return reconstruct_path(came_from, current)
            return int(g_score[(current, prev_nodes)])

        neighbours = set((current + 1j, current - 1, current - 1j, current + 1))

        try:
            if len(set((current.real, *(n.real for n in prev_nodes)))) == 1:
                neighbours = neighbours.intersection((current + 1, current - 1))
            if len(set((current.imag, *(n.imag for n in prev_nodes)))) == 1:
                neighbours = neighbours.intersection((current + 1j, current - 1j))
        except AttributeError:
            pass

        neighbours = neighbours.intersection(grid)
        for next_node in neighbours:
            tentative_g_score = g_score[(current, prev_nodes)] + grid[next_node]
            next_prev_nodes = (*prev_nodes[1:], current)
            if tentative_g_score < g_score[(next_node, next_prev_nodes)]:
                came_from[(next_node, next_prev_nodes)] = current
                g_score[(next_node, next_prev_nodes)] = tentative_g_score
                f_score[(next_node, next_prev_nodes)] = tentative_g_score + h(next_node)
                open_set.put((f_score[(next_node, next_prev_nodes)], i, next_node, next_prev_nodes))
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
puzzle.answer_a = answer


# Part b
def b(data):
    breakpoint()


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
