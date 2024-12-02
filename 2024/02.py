import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    s = 0
    for i, line in enumerate(data.splitlines()):
        numbers = [int(n) for n in line.split(" ")]
        prev = numbers[0]
        for n in numbers[1:]:
            if n >= prev:
                break
            if n < prev - 3:
                break
            prev = n
        else:
            s += 1
    for i, line in enumerate(data.splitlines()):
        numbers = [int(n) for n in line.split(" ")]
        prev = numbers[0]
        for n in numbers[1:]:
            if n <= prev:
                break
            if n > prev + 3:
                break
            prev = n
        else:
            s += 1
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    s = 0
    for i, line in enumerate(data.splitlines()):
        numbers = [int(n) for n in line.split(" ")]
        prev = numbers[0]
        hp = 1
        for n in numbers[1:]:
            if n >= prev:
                hp -= 1
                continue
            if n < prev - 3:
                hp -= 1
                continue
            prev = n
        if hp >= 0:
            s += 1
    for i, line in enumerate(data.splitlines()):
        numbers = [int(n) for n in line.split(" ")]
        prev = numbers[0]
        hp = 1
        for n in numbers[1:]:
            if n <= prev:
                hp -= 1
                continue
            if n > prev + 3:
                hp -= 1
                continue
            prev = n
        if hp >= 0:
            s += 1
    return s


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
