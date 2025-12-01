from aocd.models import Puzzle

YEAR = 2025
DAY = 1

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    c = 0
    l = 50
    for i in data.split():
        d = int(i[1:])
        l += -d if (i[0] == "L") else d
        l %= 100
        if l == 0:
            c += 1
    return c


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 1078

# Part b
def b(data):
    c = 0
    l = 50
    for i in data.split():
        d = int(i[1:])
        p = l
        l += -d if (i[0] == "L") else d
        c += abs(l) // 100
        if (l == 0) or (l < 0 and p != 0):
            c += 1
        l %= 100
    return c


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: 6)")
        assert str(example_answer) == "6"
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 6412
