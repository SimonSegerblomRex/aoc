from aocd.models import Puzzle

YEAR = 2023
DAY = 16

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def get_new_beams(pos, dir, kind):
    if kind == ".":
        return [(pos + dir, dir)]
    if kind == "|":
        if dir in (0 + 1j, 0 - 1j):
            return [(pos + 1, 1 + 0j), (pos - 1, -1 + 0j)]
        return [(pos + dir, dir)]
    if kind == "-":
        if dir in (1 + 0j, -1 + 0j):
            return [(pos - 1j, 0 - 1j), (pos + 1j, 0 + 1j)]
        return [(pos + dir, dir)]
    if kind == "/":
        new_dir = complex(-dir.imag, -dir.real)
        return [(pos + new_dir, new_dir)]
    if kind == "\\":
        new_dir = complex(dir.imag, dir.real)
        return [(pos + new_dir, new_dir)]
    raise ValueError


def create_grid(rows):
    grid = {}
    dirs = (0 + 1j, -1 + 0j, 0 - 1j, 1 + 0j)
    for i, row in enumerate(rows):
        for j, c in enumerate(row):
            pos = complex(i, j)
            for dir in dirs:
                grid[(pos, dir)] = get_new_beams(pos, dir, c)
    return grid


def count_visited(grid, height, width, start):
    beams = set([start])
    visited = set()
    while beams:
        pos, dir = beams.pop()
        if pos.real < 0 or pos.real > height - 1 or pos.imag < 0 or pos.imag > width - 1:
            continue
        visited.add((pos, dir))
        new_beams = grid[(pos, dir)]
        beams.update(new_beams)
        beams.difference_update(visited)
    return len(set(p for p, _ in visited))


def a(data):
    rows = data.splitlines()
    height, width = len(rows), len(rows[0])
    grid = create_grid(rows)
    start = (0 + 0j, 0 + 1j)
    return count_visited(grid, height, width, start)


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 8116


# Part b
def b(data):
    rows = data.splitlines()
    height, width = len(rows), len(rows[0])
    grid = create_grid(rows)
    counts = []
    for i in range(0, height):
        counts.append(count_visited(grid, height, width, (complex(i, 0), 0 + 1j)))
        counts.append(count_visited(grid, height, width, (complex(i, width - 1), 0 - 1j)))
    for j in range(0, width):
        counts.append(count_visited(grid, height, width, (complex(0, j), 1 + 0j)))
        counts.append(count_visited(grid, height, width, (complex(height - 1, j), -1 + 0j)))
    return max(counts)


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
