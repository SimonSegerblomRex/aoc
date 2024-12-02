from aocd.models import Puzzle

YEAR = 2024
DAY = 2

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def safe_a(numbers):
    prev = numbers[0]
    for n in numbers[1:]:
        if n >= prev:
            break
        if n < prev - 3:
            break
        prev = n
    else:
        return True
    prev = numbers[0]
    for n in numbers[1:]:
        if n <= prev:
            break
        if n > prev + 3:
            break
        prev = n
    else:
        return True
    return False


def a(data):
    s = 0
    for line in data.splitlines():
        numbers = [int(n) for n in line.split(" ")]
        if safe_a(numbers):
            s += 1
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 407


# Part b
def b(data):
    s = 0
    for line in data.splitlines():
        numbers = [int(n) for n in line.split(" ")]
        if safe_a(numbers):
            s += 1
            continue
        for i in range(len(numbers)):
            tmp = numbers.copy()
            tmp.pop(i)
            if safe_a(tmp):
                s += 1
                break
    return s


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 459
