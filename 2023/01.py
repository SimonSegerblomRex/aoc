import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)

if 0:
    # Part a
    def a(data):
        #data = list(map(int, data.splitlines()))
        #data = data.splitlines())
        #packs = data.split("\n\n")
        #data = re.findall("(\d+)-(\d+),(\d+)-(\d+)", data)
        #lines = iter(data.splitlines())
        """
        data = data.splitlines()
        data = (line.split(" ") for line in data)
        grid = np.vstack(
            [np.frombuffer(n.encode(), dtype=np.uint8) - ord("0") for n in data.split("\n")]
        )
        """
        lines = iter(data.splitlines())
        numbers = []
        for line in lines:
            match = re.search(".*?(\d).*(\d).*", line)
            print(line)
            try:
                numbers.append(int(match.group(1) + match.group(2)))
            except:
                match = re.search(".*(\d).*", line)
                numbers.append(int(match.group(1) + match.group(1)))
        breakpoint()
        return np.sum(numbers)
        breakpoint()


    for example in puzzle.examples:
        if example.answer_a:
            example_answer = a(example.input_data)
            print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
            assert str(example_answer) == example.answer_a
    answer = a(puzzle.input_data)
    print("a:", answer)
    puzzle.answer_a = answer


def to_number(s):
    mapp = {
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
        return mapp[s]
    except:
        return int(s)

# Part b
def b(data):
    lines = iter(data.splitlines())
    numbers = []
    try:
        for line in lines:
            match = re.search(".*?(\d|one|two|three|four|five|six|seven|eight|nine).*(\d|one|two|three|four|five|six|seven|eight|nine).*", line)
            try:
                numbers.append(int(str(to_number(match.group(1))) +  str(to_number(match.group(2)))))
            except:
                match = re.search(".*(\d|one|two|three|four|five|six|seven|eight|nine).*", line)
                numbers.append(int(str(to_number(match.group(1))) +  str(to_number(match.group(1)))))
    except:
        breakpoint()
    print(numbers)
    return np.sum(numbers)

example = """two1nine
eightwothree
abcone2threexyz
xtwone3four
4nineeightseven2
zoneight234
7pqrstsixteen
"""

tmp = b(example)
breakpoint()
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
