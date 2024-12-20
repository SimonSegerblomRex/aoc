import datetime
import queue
import re
from collections import defaultdict

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


def a_star(start, goal, walls, limit=1<<30):
    def h(node):
        return int(abs(goal.real - node.real) + abs(goal.imag - node.imag))

    open_set = queue.PriorityQueue()
    # (f_score, dummy, pos, dir)
    open_set.put((0, 0, start, 1, [start]))

    g_score = defaultdict(lambda: 1 << 30)
    g_score[(start, 1)] = 0

    f_score = defaultdict(lambda: 1 << 30)
    f_score[(start, 1)] = h(start)

    i = 1
    while not open_set.empty():
        _, _, curr_pos, curr_dir, path = open_set.get()
        if curr_pos == goal:
            return path

        dirs = [1, -1j, -1, 1j]
        for dir in dirs:
            if dir == -curr_dir:
                continue
            next_node = curr_pos + dir
            if next_node in walls:
                continue
            tentative_g_score = g_score[(curr_pos, curr_dir)] + 1
            if tentative_g_score < g_score[(next_node, dir)]:
                g_score[(next_node, dir)] = tentative_g_score
                f_score[(next_node, dir)] = tentative_g_score + h(next_node)
                open_set.put(
                    (
                        f_score[(next_node, dir)],
                        i,
                        next_node,
                        dir,
                        [*path, next_node]
                    )
                )
                i += 1


# Part a
def a(data):
    walls = set()
    track = set()
    for i, line in enumerate(data.splitlines()):
        width = len(line)
        for j, c in enumerate(line):
            if c == "#":
                walls.add(j + i*1j)
            elif c == "S":
                start = j + i*1j
            elif c == "E":
                goal = j + i*1j
    height = i + 1
    track = a_star(start, goal, walls)
    steps_from_start = dict(zip(track, range(len(track))))
    steps_to_goal = dict(zip(track, list(range(len(track)))[::-1]))
    shortest = steps_to_goal[start]
    def neihgbours(p):
        return p + 2, p - 2j, p - 2, p + 2j
    c = 0
    for pos in track:
        neigh = set(neihgbours(pos))
        neigh &= set(track)
        for n in neigh:
            if steps_from_start[pos] + 2 + steps_to_goal[n] <= shortest - 100:
                c+= 1
    return c


answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 1409


# Part b
def b(data, save):
    walls = set()
    track = set()
    for i, line in enumerate(data.splitlines()):
        width = len(line)
        for j, c in enumerate(line):
            if c == "#":
                walls.add(j + i*1j)
            elif c == "S":
                start = j + i*1j
            elif c == "E":
                goal = j + i*1j
    height = i + 1
    track = a_star(start, goal, walls)
    steps_from_start = dict(zip(track, range(len(track))))
    steps_to_goal = dict(zip(track, list(range(len(track)))[::-1]))
    shortest = steps_to_goal[start]
    def neihgbours(p):
        neighs = []
        for i in range(-20, 21):
            for j in range(-20, 21):
                if abs(i) + abs(j) > 20:
                    continue
                neighs.append(p + j + i * 1j)
        return neighs
    def manhattan(x, y):
        return int(abs(x.real - y.real) + abs(x.imag - y.imag))
    c = 0
    for pos in track:
        neigh = set(neihgbours(pos))
        neigh &= set(track)
        for n in neigh:
            if steps_from_start[pos] + manhattan(pos, n) + steps_to_goal[n] <= shortest - save:
                c+= 1
    return c


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data, 50)
        print(f"Example answer: {example_answer} (expecting: {285})")
        assert example_answer == 285
answer = b(puzzle.input_data, 100)
print("b:", answer)
puzzle.answer_b = answer
