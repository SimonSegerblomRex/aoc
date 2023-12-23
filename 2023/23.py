import re
import sys
from collections import defaultdict

import networkx as nx
import numpy as np
from aocd.models import Puzzle

YEAR = 2023
DAY = 23

puzzle = Puzzle(year=YEAR, day=DAY)

sys.setrecursionlimit(10000)


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


def b_star(paths, end, forest, slopes):
    if sum(p[-1] == end for p in paths) == len(paths):
        return paths
    new_paths = []
    for path in paths:
        current = path[-1]
        if current == end:
            new_paths.append(path)
            continue
        dirs = [0 + 1j, -1 + 0j, 0 - 1j, 1 + 0j]
        for dir in dirs:
            next_node = current + dir
            if next_node in forest:
                continue
            if next_node in path:
                continue
            if next_node in slopes:
                if dir != slopes[next_node]:
                    continue
            new_paths.append([*path, next_node])
    return b_star(new_paths, end, forest, slopes)


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

    forest.append(start - 1)
    forest.append(goal + 1)

    paths =  b_star([[start]], goal, forest, slopes)
    return max(len(p) for p in paths) - 1


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {94})")
        assert example_answer == 94
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 2086

# Part b
def find_connections(node, forest):
    dirs = [0 + 1j, -1 + 0j, 0 - 1j, 1 + 0j]
    connections = set()
    search_dirs = [dir for dir in dirs if node + dir not in forest]
    for search_dir in search_dirs:
        current = node
        next_dir = search_dir
        s = 0
        while True:
            next_node = current + next_dir
            s += 1
            next_dirs = [dir for dir in dirs if next_node + dir not in forest and dir != -next_dir]
            if len(next_dirs) != 1:
                break
            next_dir = next_dirs[0]
            current = next_node
        if 1:#len(next_dirs) > 1:
            connections.add((next_node, s))
    return connections


def create_graph(start, goal, forest):
    graph = {}
    nodes = set([start])
    while True:
        new_nodes = set()
        for node in nodes:
            if node not in graph:
                connections = find_connections(node, forest)
                connections = connections#{c for c in connections if c[0] not in graph}
                if connections:
                    graph[node] = connections
                    new_nodes |= set([n for n, _ in connections])
        if not new_nodes:
            break
        nodes = new_nodes
    return graph


def e_star(paths, goal, graph):
    if sum(p[-1] == goal for (p, _) in paths) == len(paths):
        return paths
    new_paths = []
    for (path, steps) in paths:
        current = path[-1]
        if current == goal:
            new_paths.append((path, steps))
            continue
        for (next_node, s) in graph[current]:
            if next_node in path:
                continue
            new_paths.append(([*path, next_node], steps+s))
    return e_star(new_paths, goal, graph)


def b(data):
    forest = []
    for i, line in enumerate(data.splitlines()):
        for m in re.finditer("#", line):
            j = m.start()
            forest.append(complex(i, j))
    height, width = i + 1, j + 1

    start = 0 + 1j

    goal = complex(height - 1, width - 2)

    forest.append(start - 1)
    forest.append(goal + 1)

    graph = create_graph(start, goal, forest)
    G = nx.Graph()
    for from_node, to_nodes in graph.items():
        for to_node, step in to_nodes:
            G.add_edge(from_node, to_node, weight=step)

    m = 0
    for path in nx.all_simple_paths(G, source=start, target=goal):
        m = max(m, nx.path_weight(G, path, weight="weight"))
    return m


example_answer = b(example.input_data)
print(f"Example answer: {example_answer} (expecting: {154})")
assert example_answer == 154
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 6526
