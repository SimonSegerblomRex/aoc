import networkx as nx
import numpy as np
from aocd.models import Puzzle

YEAR = 2023
DAY = 25

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    components = {}
    for line in data.splitlines():
        name, others = line.split(":")
        components[name] = others.strip().split()

    G = nx.DiGraph()
    for from_node, to_nodes in components.items():
        for to_node in to_nodes:
            G.add_edge(from_node, to_node)

    UG = G.to_undirected()
    UG.remove_edges_from(nx.minimum_edge_cut(UG))

    sizes = [len(c) for c in nx.connected_components(UG)]
    return sizes[0] * sizes[1]


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

example_answer = a(example)
print(f"Example answer: {example_answer} (expecting: {54})")
assert example_answer == 54
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 580800
