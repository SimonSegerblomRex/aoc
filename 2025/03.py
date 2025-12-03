from aocd.models import Puzzle

YEAR = 2025
DAY = 3

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, digits=2):
    s = 0
    for line in data.split():
        number = []
        start = 0
        for i in range(digits):
            view = line[start : len(line) - digits + i + 1]
            number.append(max(view))
            start += view.index(number[-1]) + 1
        s += int("".join(number))
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 16858


# Part b
for example in puzzle.examples:
    if example.answer_b:
        example_answer = a(example.input_data, 12)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = a(puzzle.input_data, 12)
print("b:", answer)
assert answer == 167549941654721
