import datetime
import queue
import re
from collections import defaultdict

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


def a_star(start, goal, walls):
    def h(node):
        return int(abs(goal.real - node.real) + abs(goal.imag - node.imag))

    open_set = queue.PriorityQueue()
    # (f_score, dummy, pos, dir)
    open_set.put((0, 0, start, 1))

    g_score = defaultdict(lambda: 1 << 30)
    g_score[(start, 1)] = 0

    f_score = defaultdict(lambda: 1 << 30)
    f_score[(start, 1)] = h(start)

    i = 1
    while not open_set.empty():
        _, _, curr_pos, curr_dir = open_set.get()
        if curr_pos == goal:
            return int(g_score[(curr_pos, curr_dir)])

        dirs = [1, -1j, -1, 1j]
        for dir in dirs:
            turning_cost = 0
            if dir != curr_dir:
                turning_cost = 0
            if dir == -curr_dir:
                continue
            next_node = curr_pos + dir
            if next_node in walls:
                continue
            tentative_g_score = g_score[(curr_pos, curr_dir)] + 1 + turning_cost
            if tentative_g_score < g_score[(next_node, dir)]:
                g_score[(next_node, dir)] = tentative_g_score
                f_score[(next_node, dir)] = tentative_g_score + h(next_node)
                open_set.put(
                    (
                        f_score[(next_node, dir)],
                        i,
                        next_node,
                        dir,
                    )
                )
                i += 1


# Part a
def a(data, width=7, height=7, stop=12):
    obstacles = set()
    for n, line in enumerate(data.splitlines()):
        if n == stop:
            break
        j, i = line.split(",")
        obstacles.add(int(j) + int(i)*1j)
    for i in (-1, height):
        for j in range(-1, width):
            obstacles.add(j + i*1j)
    for j in (-1, width):
        for i in range(-1, height):
            obstacles.add(j + i*1j)
    if 0:
        for i in range(-1, height + 1):
            for j in range(-1, width + 1):
                p = j + i*1j
                if p in obstacles:
                    print("#", end="")
                else:
                    print(".", end="")
            print()
    return a_star(0j, width - 1 + (height - 1)* 1j, obstacles)


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {22})")
        assert example_answer == 22
answer = a(puzzle.input_data, width=71, height=71, stop=1024)
print("a:", answer)
assert answer == 234


# Part b
def b(data, width=7, height=7, guess=13):
    obstacles = []
    for n, line in enumerate(data.splitlines()):
        j, i = line.split(",")
        obstacles.append(int(j) + int(i)*1j)
    stop = guess
    while True:
        answer = a(data, width=width, height=height, stop=stop)
        if answer is None:
            break
        stop += 1
    stop -= 1
    return f"{int(obstacles[stop].real)},{int(obstacles[stop].imag)}"


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: 6,1)")
        assert example_answer == "6,1"
answer = b(puzzle.input_data, width=71, height=71, guess=3000)
print("b:", answer)
puzzle.answer_b = answer
