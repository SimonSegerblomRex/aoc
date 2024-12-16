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
                turning_cost = 1000
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
def a(data):
    walls = set()
    for i, line in enumerate(data.splitlines()):
        for j, c in enumerate(line):
            if c == "#":
                walls.add(j + i*1j)
            elif c == "S":
                start = j + i*1j
            elif c == "E":
                goal = j + i*1j

    return a_star(start, goal, walls)


answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


def a_star_modified(start, goal, walls):
    def h(node):
        return 0#int(abs(goal.real - node.real) + abs(goal.imag - node.imag))

    open_set = []
    # (f_score, dummy, pos, dir, path)
    open_set.append((0, 0, start, 1, [start]))

    g_score = defaultdict(lambda: 1 << 30)
    g_score[(start, 1)] = 0

    f_score = defaultdict(lambda: 1 << 30)
    f_score[(start, 1)] = h(start)

    i = 1
    good = set()
    best_score = 1 << 30
    tmp = []
    while open_set:
        _, _, curr_pos, curr_dir, path = open_set.pop(0)
        if curr_pos == goal:
            tmp.append((int(g_score[(curr_pos, curr_dir)]), path))
            if int(g_score[(curr_pos, curr_dir)]) < best_score:
                best_score = int(g_score[(curr_pos, curr_dir)])
            if int(g_score[(curr_pos, curr_dir)]) <= best_score:
                good |= set(path)

        dirs = [1, -1j, -1, 1j]
        for dir in dirs:
            turning_cost = 0
            if dir != curr_dir:
                turning_cost = 1000
            if dir == -curr_dir:
                continue
            next_node = curr_pos + dir
            if next_node in walls:
                continue
            tentative_g_score = g_score[(curr_pos, curr_dir)] + 1 + turning_cost
            if tentative_g_score <= g_score[(next_node, dir)]:
                g_score[(next_node, dir)] = tentative_g_score
                f_score[(next_node, dir)] = tentative_g_score + h(next_node)
                open_set.append(
                    (
                        f_score[(next_node, dir)],
                        i,
                        next_node,
                        dir,
                        [*path, next_node]
                    )
                )
                i += 1

    if 1:
        height = int(max(n.imag for n in walls)) + 1
        width = int(max(n.real for n in walls)) + 1
        for i in range(height):
            for j in range(width):
                p =  j + i * 1j
                if p in walls:
                    print("#", end="")
                elif p in good:
                    print("O", end="")
                else:
                    print(".", end="")
            print()

    best_score = min(s for s, _ in tmp)
    gg = set()
    for s, p in tmp:
        if s == best_score:
            gg |= set(p)
    return len(gg)
    breakpoint()
    return len(good)


# Part b
def b(data):
    walls = set()
    for i, line in enumerate(data.splitlines()):
        for j, c in enumerate(line):
            if c == "#":
                walls.add(j + i*1j)
            elif c == "S":
                start = j + i*1j
            elif c == "E":
                goal = j + i*1j

    return a_star_modified(start, goal, walls)


answer = b(puzzle.examples[0].input_data)
print("b:", answer)
assert answer == 45
example = """#################
#...#...#...#..E#
#.#.#.#.#.#.#.#.#
#.#.#.#...#...#.#
#.#.#.#.###.#.#.#
#...#.#.#.....#.#
#.#.#.#.#.#####.#
#.#...#.#.#.....#
#.#.#####.#.###.#
#.#.#.......#...#
#.#.###.#####.###
#.#.#...#.....#.#
#.#.#.#####.###.#
#.#.#.........#.#
#.#.#.#########.#
#S#.............#
#################"""
answer = b(example)
print("b:", answer)
assert answer == 64
answer = b(puzzle.input_data)
print("b:", answer)
assert answer < 700
puzzle.answer_b = answer
