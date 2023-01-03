from aocd.models import Puzzle

YEAR = 2022
DAY = 1

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    packs = data.split("\n\n")
    calories_total = []
    for pack in packs:
        calories_total.append(sum([int(food) for food in pack.split("\n")]))
    return max(calories_total)


example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 24000
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 69281


# Part b
def b(data):
    packs = data.split("\n\n")
    calories_total = []
    for pack in packs:
        calories_total.append(sum([int(food) for food in pack.split("\n")]))
    return sum(sorted(calories_total)[-3:])


example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 45000
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 201524
