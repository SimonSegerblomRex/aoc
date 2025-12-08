from aocd.models import Puzzle

YEAR = 2025
DAY = 8

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def distance(x0, y0, z0, x1, y1, z1):
    return (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2


def a(data, stop=10):
    coords = set()
    for line in data.splitlines():
        coords.add(tuple(map(int, line.split(","))))
    distances = {}
    for c0 in coords:
        for c1 in coords:
            if c0 == c1:
                continue
            if (c1, c0) not in distances:
                distances[(c0, c1)] = distance(*c0, *c1)
    distances = dict(sorted(distances.items(), key=lambda t: t[1]))
    circuits = [{coord} for coord in coords]
    circuit_map = {coord: circuit for coord, circuit in zip(coords, circuits)}
    for i, (c0, c1) in enumerate(distances):
        if i == stop:
            break
        if circuit_map[c0] == circuit_map[c1]:
            # Do nothing
            continue
        # Combine
        circuit_map[c0] |= circuit_map[c1]
        circuits.remove(circuit_map[c1])
        for c in circuit_map[c1]:
            circuit_map[c] = circuit_map[c0]
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
            if (c1, c0) not in distances:
                distances[(c0, c1)] = distance(*c0, *c1)
    distances = dict(sorted(distances.items(), key=lambda t: t[1]))
    circuits = [{coord} for coord in coords]
    circuit_map = {coord: circuit for coord, circuit in zip(coords, circuits)}
    for c0, c1 in distances:
        if circuit_map[c0] == circuit_map[c1]:
            # Do nothing
            continue
        # Combine
        circuit_map[c0] |= circuit_map[c1]
        circuits.remove(circuit_map[c1])
        for c in circuit_map[c1]:
            circuit_map[c] = circuit_map[c0]
        if len(circuits[0]) == len(coords):
            return c0[0] * c1[0]


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 724454082
