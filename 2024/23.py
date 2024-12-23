from collections import defaultdict

import networkx as nx
from aocd.models import Puzzle

YEAR = 2024
DAY = 23

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    connections = defaultdict(list)
    for line in data.split():
        c0, c1 = line.split("-")
        connections[c0].append(c1)
        connections[c1].append(c0)
    triplets = set()
    for c0, conns0 in connections.items():
        for c1, conns1 in connections.items():
            if c0 == c1:
                continue
            if c1 not in conns0:
                continue
            if c0 not in conns1:
                continue
            for c2 in set(conns0) & set(conns1):
                triplets.add(tuple(sorted([c0, c1, c2])))
    return sum(1 for t in triplets if "t" in "".join(chr0 for chr0, _ in t))


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {7})")
        assert example_answer == 7
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 1098


# Part b
def b(data):
    connections = defaultdict(list)
    for line in data.split():
        c0, c1 = line.split("-")
        connections[c0].append(c1)
        connections[c1].append(c0)
    G = nx.Graph()
    for from_node, to_nodes in connections.items():
        for to_node in to_nodes:
            G.add_edge(from_node, to_node)
    cliques = ((len(clique), clique) for clique in nx.find_cliques(G))
    return  ",".join(sorted(sorted(cliques)[-1][1]))


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == "ar,ep,ih,ju,jx,le,ol,pk,pm,pp,xf,yu,zg"
