import numpy as np
from aocd.models import Puzzle

YEAR = 2024
DAY = 5

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    rules, updates = data.split("\n\n")
    rules = [list(map(int, rule.split("|"))) for rule in rules.splitlines()]
    updates = [list(map(int, update.split(","))) for update in updates.splitlines()]
    in_order = []
    for update in updates:
        for i in range(len(update) - 1):
            if [update[i], update[i + 1]] not in rules:
                break
        else:
            in_order.append(update)
    s = 0
    for update in in_order:
        s += update[len(update) // 2]
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 4957

# Part b
def b(data):
    rules, updates = data.split("\n\n")
    rules = [list(map(int, rule.split("|"))) for rule in rules.splitlines()]
    updates = [list(map(int, update.split(","))) for update in updates.splitlines()]
    in_order = []
    for update in updates:
        for i in range(len(update) - 1):
            if [update[i], update[i + 1]] not in rules:
                break
        else:
            in_order.append(update)
    s = 0
    for update in updates:
        if update in in_order:
            continue
        while True:
            for i in range(len(update) - 1):
                if [update[i], update[i + 1]] not in rules:
                    break
            else:
                break
            for i in range(len(update) - 1):
                if [update[i], update[i + 1]] not in rules:
                    update[i], update[i + 1] = update[i + 1], update[i]
        s += update[len(update) // 2]
    return s


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 6938
