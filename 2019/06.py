from aocd.models import Puzzle

YEAR = 2019
DAY = 6

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    orbits = data.splitlines()
    orbit_map = {}
    for orbit in orbits:
        a, b = orbit.split(")")
        orbit_map[b] = a
    def count(o):
        n = 0
        if o in orbit_map:
            n = 1
            n += count(orbit_map[o])
        return n
    return sum(count(o) for o in orbit_map)


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 315757

# Part b
def b(data):
    orbits = data.splitlines()
    orbit_map = {}
    for orbit in orbits:
        a, b = orbit.split(")")
        orbit_map[b] = a
    def parents(o):
        p = []
        if o in orbit_map:
            p.append(orbit_map[o])
            p.extend(parents(orbit_map[o]))
        return p
    you = parents("YOU")
    san = parents("SAN")
    for i, p in enumerate(you):
        if p in san:
            return i + san.index(p)


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 481
