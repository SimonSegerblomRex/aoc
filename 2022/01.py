from aocd.models import Puzzle

YEAR = 2022
DAY = 1

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(input):
    packs = input.split("\n\n")
    calories_total = []
    for pack in packs:
        calories_total.append(sum([int(food) for food in pack.split("\n")]))
    return max(calories_total)

ANSWER_EXAMPLE = 24000
assert a(puzzle.example_data) == ANSWER_EXAMPLE
answer = a(puzzle.input_data)
print("a:", answer)
#puzzle.answer_a = answer


# Part b
def b(input):
    packs = input.split("\n\n")
    calories_total = []
    for pack in packs:
        calories_total.append(sum([int(food) for food in pack.split("\n")]))
    return sum(sorted(calories_total)[-3:])

ANSWER_EXAMPLE = 45000
assert b(puzzle.example_data) == ANSWER_EXAMPLE
answer = b(puzzle.input_data)
print("b:", answer)
#puzzle.answer_b = answer
