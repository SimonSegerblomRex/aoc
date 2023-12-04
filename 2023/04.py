from aocd.models import Puzzle

YEAR = 2023
DAY = 4

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    p = 0
    for n, line in enumerate(data.splitlines(), 1):
        l1, l2 = line.split("|")
        l1 = l1.split(":")[1]
        l1 = [int(e) for e in l1.strip().split(" ") if e != ""]
        l2 = [int(e) for e in l2.strip().split(" ") if e != ""]
        winning_numbers = list(set(l2).intersection(l1))
        if winning_numbers:
            p += 2 ** (len(winning_numbers) - 1)
    return p


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 20407


# Part b
def b(data):
    lines = data.splitlines()
    copies = [0] * (len(lines) + 1)
    p = len(lines)
    for n, line in enumerate(lines, 1):
        l1, l2 = line.split("|")
        l1 = l1.split(":")[1]
        l1 = [int(e) for e in l1.strip().split(" ") if e != ""]
        l2 = [int(e) for e in l2.strip().split(" ") if e != ""]
        winning_numbers = list(set(l2).intersection(l1))
        for i in range(n + 1, n + len(winning_numbers) + 1):
            copies[i] += 1 + copies[n]
    return p + sum(copies)


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 23806951
