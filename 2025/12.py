from aocd.models import Puzzle

YEAR = 2025
DAY = 12

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def rot90(width, height, coords):
    return (height, width, frozenset((-y + height - 1, x) for x, y in coords))


def parse_shape(shape):
    coords = set()
    for i, line in enumerate(shape.splitlines()[1:]):
        for j, c in enumerate(line):
            if c == "#":
                coords.add((j, i))
    return (j + 1, i + 1, frozenset(coords))


def print_shape(width, height, coords):
    for i in range(height):
        for j in range(width):
            if (j, i) in coords:
                print("#", end="")
            else:
                print(" ", end="")
        print()


def with_rotations(shapes):
    out = []
    for shape in shapes:
        out.append(set((
            shape,
            rot90(*shape),
            rot90(*rot90(*shape)),
            rot90(*rot90(*rot90(*shape))),
        )))
    return out


def a(data):
    *shapes, regions = data.split("\n\n")
    shapes = [parse_shape(shape) for shape in shapes]
    shapes = with_rotations(shapes)
    s = 0
    for region in regions.splitlines():
        size, quantities = region.split(":")
        width, height = list(map(int, size.split("x")))
        quantities = list(map(int, quantities.strip().split()))
        breakpoint()


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
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
