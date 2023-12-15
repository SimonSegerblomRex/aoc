from collections import defaultdict

from aocd.models import Puzzle

YEAR = 2023
DAY = 15

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def holiday_hash(s):
    v = 0
    for c in s:
        v += ord(c)
        v *= 17
        v %= 256
    return v


def a(data):
    steps = data.split(",")
    return sum(holiday_hash(s) for s in steps)


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 516804


# Part b
def b(data):
    boxes = defaultdict(dict)
    for step in data.split(","):
        if "-" in step:
            label = step.split("-")[0]
            for _, box in boxes.items():
                if label in box:
                    del box[label]
        elif "=" in step:
            label, focal_length = step.split("=")
            boxes[holiday_hash(label)][label] = int(focal_length)
    s = 0
    for box_number, box in boxes.items():
        for slot_number, lens in enumerate(box.items(), 1):
            s += (box_number + 1) * slot_number * lens[1]
    return s


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 231844
