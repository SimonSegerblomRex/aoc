from aocd.models import Puzzle

YEAR = 2025
DAY = 8

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def distance(x0, y0, z0, x1, y1, z1):
    return (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2


def a(data, stop=None):
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
    circuits = {coord: {coord} for coord in coords}
    for i, (c0, c1) in enumerate(distances):
        # Stop criterion for a
        if i == stop:
            break
        if circuits[c0] is circuits[c1]:
            # Do nothing
            continue
        # Combine
        circuits[c0] |= circuits[c1]
        for coord in circuits[c1]:
            circuits[coord] = circuits[c0]
        # Stop criterion for b
        if len(circuits[c0]) == len(coords):
            return c0[0] * c1[0]
    circuits = {frozenset(c) for c in circuits.values()}
    s = 1
    for i, circuit in enumerate(sorted(circuits, key=lambda c: len(c), reverse=True)):
        s *= len(circuit)
        if i > 1:
            break
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data, 10)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data, 1000)
print("a:", answer)
assert answer == 66912


# Part b
for example in puzzle.examples:
    if example.answer_b:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = a(puzzle.input_data)
print("b:", answer)
assert answer == 724454082
