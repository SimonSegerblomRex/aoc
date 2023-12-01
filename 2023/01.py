import datetime
import re

from aocd.models import Puzzle

YEAR = 2023
DAY = 1

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    lines = iter(data.splitlines())
    numbers = []
    for line in lines:
        digits = re.findall(r"(\d)", line)
        numbers.append(int(digits[0] + digits[-1]))
    return sum(numbers)


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 56049


def to_digit(s):
    string_map = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
    }
    try:
        return string_map[s]
    except KeyError:
        return int(s)


# Part b
def b(data):
    lines = iter(data.splitlines())
    numbers = []
    for line in lines:
        digits = re.findall(
            r"(?=(\d|one|two|three|four|five|six|seven|eight|nine))", line
        )
        numbers.append(int(str(to_digit(digits[0])) + str(to_digit(digits[-1]))))
    return sum(numbers)


example = """two1nine
eightwothree
abcone2threexyz
xtwone3four
4nineeightseven2
zoneight234
7pqrstsixteen
"""

assert b(example) == 281
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 54530
