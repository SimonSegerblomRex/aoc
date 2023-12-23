import datetime
import re
import queue
from collections import defaultdict

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


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


def a_star(start, end, forest, slopes):
    def h(node):
        return int(abs(end.real - node.real) + abs(end.imag - node.imag))

    open_set = queue.PriorityQueue()
    open_set.put((0, 0, start))

    g_score = defaultdict(lambda: 0)
    g_score[start] = 0

    f_score = defaultdict(lambda: 0)
    f_score[start] = -h(start)

    came_from = {}
    came_from[start] = -1 + 1j # FIXME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    i = 2

    while not open_set.empty():
        _, _, current = open_set.get()
        if current == end:
            tmp = reconstruct_path(came_from, current)
            debug_print(forest, tmp)
            print(len(tmp))
            breakpoint()

            return int(g_score[current])

        dirs = [0 + 1j, -1 + 0j, 0 - 1j, 1 + 0j]
        for dir in dirs:
            next_node = current + dir
            if next_node in forest:
                continue
            if came_from[current] == next_node:
                continue
            if next_node in slopes:
                if dir != slopes[next_node]:
                    continue
            tentative_g_score = g_score[current] + h(next_node)
            if tentative_g_score > g_score[next_node]:
                g_score[next_node] = tentative_g_score
                f_score[next_node] = tentative_g_score + h(next_node)
                came_from[next_node] = current
                open_set.put(
                    (
                        -f_score[next_node],
                        i,
                        next_node,
                    )
                )
                i += 1


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

    tmp =  a_star(start, goal, forest, slopes)
    breakpoint()


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
