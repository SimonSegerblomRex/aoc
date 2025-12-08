from aocd.models import Puzzle

YEAR = 2025
DAY = 8

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def distance(x0, y0, z0, x1, y1, z1):
    return (x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2

def a(data, stop=10):
    coords = set()
    for line in data.splitlines():
        coords.add(tuple(map(int, line.split(","))))
    distances = {}
    for c0 in coords:
        for c1 in coords:
            if c0 == c1:
                continue
            #distances[tuple(sorted((c0, c1)))] = distance(*c0, *c1)
            if (c1, c0) not in distances:
                distances[(c0, c1)] = distance(*c0, *c1)
    distances = dict(sorted(distances.items(), key=lambda t: t[1]))
    circuits = []
    circuit_map = {}
    i = 0
    for c0, c1 in distances:
        if i >= stop:
            break
        if c0 in circuit_map and c1 in circuit_map:
            if circuit_map[c0] == circuit_map[c1]:
                # Do nothing
                i += 1
                continue
            # Combine
            circuit = circuit_map[c1]
            circuits.remove(circuit)
            for c in circuit:
                circuit_map[c] = circuit_map[c0]
                circuit_map[c0].add(c)
            i += 1
        elif c0 in circuit_map:
            # c1 new
            assert c1 not in circuit_map
            circuit_map[c0].add(c1)
            circuit_map[c1] = circuit_map[c0]
            i += 1
        elif c1 in circuit_map:
            # c0 new
            assert c0 not in circuit_map
            circuit_map[c1].add(c0)
            circuit_map[c0] = circuit_map[c1]
            i += 1
        else:
            circuit = set((c0, c1))
            circuits.append(circuit)
            circuit_map[c0] = circuit
            circuit_map[c1] = circuit
            i += 1
    sizes = sorted([len(c) for c in circuits], reverse=True)
    s = 1
    for i in range(3):
        s *= sizes[i]
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data, 1000)
print("a:", answer)
assert answer == 66912


# Part b
def b(data):
    coords = set()
    for line in data.splitlines():
        coords.add(tuple(map(int, line.split(","))))
    distances = {}
    for c0 in coords:
        for c1 in coords:
            if c0 == c1:
                continue
            #distances[tuple(sorted((c0, c1)))] = distance(*c0, *c1)
            if (c1, c0) not in distances:
                distances[(c0, c1)] = distance(*c0, *c1)
    distances = dict(sorted(distances.items(), key=lambda t: t[1]))
    circuits = []
    circuit_map = {}
    for c0, c1 in distances:
        if c0 in circuit_map and c1 in circuit_map:
            if circuit_map[c0] == circuit_map[c1]:
                # Do nothing
                continue
            # Combine
            circuit = circuit_map[c1]
            circuits.remove(circuit)
            for c in circuit:
                circuit_map[c] = circuit_map[c0]
                circuit_map[c0].add(c)
        elif c0 in circuit_map:
            # c1 new
            assert c1 not in circuit_map
            circuit_map[c0].add(c1)
            circuit_map[c1] = circuit_map[c0]
        elif c1 in circuit_map:
            # c0 new
            assert c0 not in circuit_map
            circuit_map[c1].add(c0)
            circuit_map[c0] = circuit_map[c1]
        else:
            circuit = set((c0, c1))
            circuits.append(circuit)
            circuit_map[c0] = circuit
            circuit_map[c1] = circuit
        if len(circuits[0]) == len(coords):
            return c0[0] * c1[0]


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
