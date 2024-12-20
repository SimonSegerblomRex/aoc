from aocd.models import Puzzle

YEAR = 2024
DAY = 1

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    l1 = []
    l2 = []
    for line in data.splitlines():
        line = line.split(" ")
        l1.append(int(line[0]))
        l2.append(int(line[-1]))
    l1 = sorted(l1)
    l2 = sorted(l2)
    s = 0
    for i in range(len(l1)):
        s += abs(l1[i] - l2[i])
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 765748


# Part b
def b(data):
    l1 = []
    l2 = []
    for line in data.splitlines():
        line = line.split(" ")
        l1.append(int(line[0]))
        l2.append(int(line[-1]))
    s = 0
    for i in range(len(l1)):
        c = 0
        for n in l2:
            if n == l1[i]:
                c += 1
        s += c * l1[i]
    return s


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 27732508
