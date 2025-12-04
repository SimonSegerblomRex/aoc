from aocd.models import Puzzle

YEAR = 2025
DAY = 4

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    pos = set()
    for j, line in enumerate(data.split()):
        for i, ch in enumerate(line):
            if ch == "@":
                pos.add(i + j * 1j)
    c = 0
    for p in pos:
        neighbours = {
            p + 1,
            p + 1 - 1j,
            p - 1j,
            p - 1 - 1j,
            p - 1,
            p - 1 + 1j,
            p + 1j,
            p + 1 + 1j,
        }
        if len(neighbours & pos) < 4:
            c += 1
    return c


answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 1549

# Part b
def b(data):
    pos = set()
    for j, line in enumerate(data.split()):
        for i, ch in enumerate(line):
            if ch == "@":
                pos.add(i + j * 1j)
    c = 0
    while True:
        new_pos = set()
        old_c = c
        for p in pos:
            neighbours = {
                p + 1,
                p + 1 - 1j,
                p - 1j,
                p - 1 - 1j,
                p - 1,
                p - 1 + 1j,
                p + 1j,
                p + 1 + 1j,
            }
            if len(neighbours & pos) < 4:
                c += 1
            else:
                new_pos.add(p)
        pos = new_pos
        if old_c == c:
            break
    return c


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 8887
