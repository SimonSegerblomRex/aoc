from aocd.models import Puzzle

YEAR = 2023
DAY = 18

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def calculate_area(corners):
    edge = 0
    area = 0
    for i in range(1, len(corners)):
        edge += abs(corners[i - 1] - corners[i])
        area += int(corners[i].real) * (int(corners[i].imag) - int(corners[i - 1].imag))
    area = abs(area)
    return area + int(edge) // 2 + 1


def a(data):
    corners = [0 + 0]
    dirs = {
        "R": 0 + 1j,
        "U": -1 + 0j,
        "L": 0 - 1j,
        "D": 1 + 0j,
    }
    curr = corners[0]
    for line in data.splitlines():
        dir, steps, _ = line.split(" ")
        curr += dirs[dir] * int(steps)
        corners.append(curr)
    corners.append(corners[0])
    return calculate_area(corners)


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 48400


# Part b
def b(data):
    corners = [0 + 0]
    dirs = {
        "0": 0 + 1j,
        "3": -1 + 0j,
        "2": 0 - 1j,
        "1": 1 + 0j,
    }
    curr = corners[0]
    for line in data.splitlines():
        _, _, hex_code = line.split(" ")
        steps = int(hex_code[2:7], 16)
        dir = hex_code[7]
        curr += dirs[dir] * int(steps)
        corners.append(curr)
    corners.append(corners[0])
    return calculate_area(corners)


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 72811019847283
