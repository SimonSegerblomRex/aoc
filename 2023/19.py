import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    workflows_input, ratings_input = data.split("\n\n")
    workflows = {}
    for line in workflows_input.splitlines():
        name, rules = line.split("{")
        rules = rules.strip("}").split(",")
        workflows[name] = rules
    s = 0
    for line in ratings_input.splitlines():
        part_spec = {}
        for entry in line.strip("{}").split(","):
            category, value = entry.split("=")
            part_spec[category] = int(value)
        curr_workflow = "in"
        while True:
            rules = workflows[curr_workflow]
            for rule in rules:
                if "<" in rule:
                    c, v = rule.split("<")
                    v, dest = v.split(":")
                    if part_spec[c] < int(v):
                        curr_workflow = dest
                        break
                elif ">" in rule:
                    c, v = rule.split(">")
                    v, dest = v.split(":")
                    if part_spec[c] > int(v):
                        curr_workflow = dest
                        break
                else:
                    curr_workflow = rule
            if curr_workflow == "A":
                s += sum(part_spec.values())
                break
            elif curr_workflow == "R":
                break
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
    breakpoint()


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
