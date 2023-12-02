import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    try:
        lines = iter(data.splitlines())
        good_games = []
        for line in lines:
            p1, p2 = line.split(":")
            game_id = int(p1.split(" ")[1])
            subsets = p2.split(";")
            for cset in subsets:
                nc = cset.split(",")
                nn = {"red": 0, "green": 0, "blue": 0}
                for e in nc:
                    n, c = e.strip().split(" ")
                    nn[c] += int(n)
                if nn["red"] > 12 or nn["green"] > 13 or nn["blue"] > 14:
                    print()
                    break
            else:
                good_games.append(game_id)
        return sum(good_games)
        breakpoint()
    except:
        breakpoint()


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
    print(data)
    breakpoint()


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
