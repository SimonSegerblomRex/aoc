from functools import cache

from aocd.models import Puzzle

YEAR = 2024
DAY = 19

puzzle = Puzzle(year=YEAR, day=DAY)



# Part a
def a(data):
    patterns, designs = data.split("\n\n")
    patterns = patterns.split(", ")
    designs = designs.split()
    @cache
    def check(design):
        if design in patterns:
            return True
        for i in range(len(design)):
            if design[:i] in patterns:
                if check(design[i:]):
                    return True
        return False
    return sum(check(design) for design in designs)


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 238


# Part b
def b(data):
    patterns, designs = data.split("\n\n")
    patterns = patterns.split(", ")
    designs = designs.split()
    @cache
    def count(design):
        c = 1 if design in patterns else 0
        for i in range(len(design)):
            if design[:i] in patterns:
                c += count(design[i:])
        return c
    return sum(count(design) for design in designs)


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {16})")
        assert example_answer == 16
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 635018909726691
