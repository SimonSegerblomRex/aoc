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
            if c0 != c1 and (c1, c0) not in distances:
                distances[(c0, c1)] = distance(*c0, *c1)
    distances = sorted(distances, key=lambda k: distances[k])[:stop]
    circuits = {coord: {coord} for coord in coords}
    for c0, c1 in distances:
        circuits[c0] |= circuits[c1]
        for coord in circuits[c1]:
            circuits[coord] = circuits[c0]
        if len(circuits[c0]) == len(coords):
            return c0[0] * c1[0]
    circuits = {frozenset(circuit) for circuit in circuits.values()}
    sizes = sorted(map(len, circuits), reverse=True)
    return sizes[0] * sizes[1] * sizes[2]


assert (answer := a(puzzle.examples[0].input_data, 10)) == 40, answer
answer = a(puzzle.input_data, 1000)
print("a:", answer)
assert answer == 66912


# Part b
assert (answer := a(puzzle.examples[0].input_data)) == 25272, answer
answer = a(puzzle.input_data)
print("b:", answer)
assert answer == 724454082
