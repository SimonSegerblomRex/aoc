import re

from aocd.models import Puzzle

YEAR = 2025
DAY = 6

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    numbers = re.findall(r"\d+", data)
    operators = data.splitlines()[-1].split()
    nbr_problems = len(operators)
    s = 0
    for i, operator in enumerate(operators):
        s += eval(f"{operator}".join(numbers[i::nbr_problems]))
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 5335495999141


# Part b
def b(data):
    *data, operators = data.splitlines()
    numbers = ["".join(line[i] for line in data).strip() for i in range(len(data[0]))]
    start_idx = [i for i, operator in enumerate(operators) if operator != " "]
    stop_idx = [idx - 1 for idx in start_idx[1:]]
    stop_idx.append(len(operators))
    s = 0
    for start, stop, operator in zip(start_idx, stop_idx, operators.split()):
        s += eval(f"{operator}".join(numbers[start:stop]))
    return s


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 10142723156431
