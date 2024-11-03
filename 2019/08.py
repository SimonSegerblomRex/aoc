import numpy as np
from aocd.models import Puzzle

YEAR = 2019
DAY = 8

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    image = np.array(list(map(int, data)))
    image.shape = (-1, 6, 25)
    zeros = np.sum(image == 0, axis=(1, 2))
    layer_idx = zeros.argmin()
    return (image[layer_idx, ...] == 1).sum() * (image[layer_idx, ...] == 2).sum()

answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 2080

# Part b
def b(data):
    print(data)
    breakpoint()


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
