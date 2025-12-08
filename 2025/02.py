from aocd.models import Puzzle

YEAR = 2025
DAY = 2

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    s = 0
    for line in data.split(","):
        lo, hi = line.split("-")
        for n in range(int(lo), int(hi) + 1):
            n = str(n)
            if len(n) % 2:
                continue
            if n[: len(n) // 2] == n[len(n) // 2 :]:
                s += int(n)
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 44854383294


# found "batched" in the itertools documentation...
# ...but only had python3.9 on my Chromebook...
def batched(s, n):
    return zip(*[iter(s)] * n)


# Part b
def b(data):
    s = 0
    for line in data.split(","):
        lo, hi = line.split("-")
        for n in range(int(lo), int(hi) + 1):
            n = str(n)
            for c in range(1, len(n) // 2 + 1):
                if len(n) % c:
                    continue
                if len(set(batched(n, c))) == 1:
                    s += int(n)
                    break
    return s


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 55647141923
