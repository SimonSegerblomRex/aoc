import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    components = {}
    for line in data.splitlines():
        name, others = line.split(":")
        components[name] = others.strip().split()

    import networkx as nx
    G = nx.DiGraph()
    for from_node, to_nodes in components.items():
        for to_node in to_nodes:
            G.add_edge(from_node, to_node)

    UG = G.to_undirected()

    if 1:
        UG.remove_edge("mrd", "rjs")
        UG.remove_edge("gmr", "ntx")
        UG.remove_edge("ncg", "gsk")

    if 0:
        nx.draw_networkx(UG)
        import matplotlib.pyplot as plt
        plt.show()

    subgraphs = [UG.subgraph(c).copy() for c in nx.connected_components(UG)]
    return len(subgraphs[0].nodes) * len(subgraphs[1].nodes)


example = """jqt: rhn xhk nvd
rsh: frs pzl lsr
xhk: hfx
cmg: qnr nvd lhk bvb
rhn: xhk bvb hfx
bvb: xhk hfx
pzl: lsr hfx nvd
qnr: nvd
ntq: jqt hfx bvb xhk
nvd: lhk
lsr: lhk
rzs: qnr cmg lsr rsh
frs: qnr lhk lsr"""

#example_answer = a(example)
#print(f"Example answer: {example_answer} (expecting: {54})")
#assert example_answer == 54
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
