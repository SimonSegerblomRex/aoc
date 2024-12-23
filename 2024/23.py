from collections import defaultdict

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
                triplets.add(frozenset([c0, c1, c2]))
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
    good = []
    for c0, conns0 in connections.items():
        clique = set([c0])
        tmp = []
        for c1 in conns0:
            clique.add(c1)
            tmp.append(clique.copy())
        for clique in tmp:
            ok = True
            for c0 in clique:
                for c1 in clique:
                    if c0 == c1:
                        continue
                    if c0 not in connections[c1]:
                        ok = False
                        break
            if ok:
                good.append(clique)
    cliques = ((len(clique), clique) for clique in good)
    # Alternative solution using networkx:
    # import networkx as nx
    # G = nx.Graph(connections)
    # cliques = ((len(clique), clique) for clique in nx.find_cliques(G))
    return ",".join(sorted(sorted(cliques)[-1][1]))


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == "ar,ep,ih,ju,jx,le,ol,pk,pm,pp,xf,yu,zg"
