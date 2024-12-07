import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


def evaluate(numbers, current, answer):
    if len(numbers) == 1:
        return current == answer
    if (evaluate(numbers[1:], current + numbers[1], answer)
        or evaluate(numbers[1:], current * numbers[1], answer)):
        return True
    return False


# Part a
def a(data):
    s = 0
    for line in data.splitlines():
        answer, numbers = line.split(":")
        answer = int(answer)
        numbers = [int(n) for n in numbers.strip().split()]
        if evaluate(numbers, numbers[0], answer):
            s += answer
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


def evaluate_b(numbers, current, answer):
    if len(numbers) == 1:
        return current == answer
    if (evaluate(numbers[1:], current + numbers[1], answer)
        or evaluate(numbers[1:], current * numbers[1], answer)
        or evaluate(numbers[1:], int(str(current) + str(numbers[1])), answer)):
        return True
    return False


# Part b
def b(data):
    s = 0
    for line in data.splitlines():
        answer, numbers = line.split(":")
        answer = int(answer)
        numbers = [int(n) for n in numbers.strip().split()]
        if evaluate_b(numbers, numbers[0], answer):
            s += answer
    return s


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
