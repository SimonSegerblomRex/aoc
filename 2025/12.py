from aocd.models import Puzzle

YEAR = 2025
DAY = 12

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def parse_shape(shape):
    coords = set()
    for i, line in enumerate(shape.splitlines()[1:]):
        for j, c in enumerate(line):
            if c == "#":
                coords.add((j, i))
    return frozenset(coords)


def a(data):
    *shapes, regions = data.split("\n\n")
    shapes = [parse_shape(shape) for shape in shapes]
    shape_areas = [len(shape) for shape in shapes]
    s = 0
    for region in regions.splitlines():
        size, quantities = region.split(":")
        width, height = map(int, size.split("x"))
        quantities = map(int, quantities.strip().split())
        rectangle_area = width * height
        minimum_area_required = sum(
            shape_area * quantity
            for shape_area, quantity in zip(shape_areas, quantities)
        )
        if minimum_area_required <= rectangle_area:
            s += 1
    return s


example_answer = a(puzzle.examples[0].input_data)
print(f"Example answer: {example_answer} (expecting: {puzzle.examples[0].answer_a})")
# assert example_answer == int(puzzle.examples[0].answer_a)
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 589
