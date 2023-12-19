import datetime
import re
from collections import defaultdict

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
assert answer == 350678


# Part b
def b(data):
    workflows_input, ratings_input = data.split("\n\n")
    workflows = {}
    for line in workflows_input.splitlines():
        name, rules = line.split("{")
        rules = rules.strip("}").split(",")
        workflows[name] = rules
    s = 0
    for line in ratings_input.splitlines():
        curr_workflow = "in"
        hard_min_limits = {"x": 1, "m": 1, "a": 1, "s": 1}
        hard_max_limits = {"x": 4000, "m": 4000, "a": 4000, "s": 4000}
        soft_min_limits = defaultdict(lambda: 0)
        soft_max_limits = defaultdict(lambda: 0)
        while True:
            rules = workflows[curr_workflow]
            for rule in rules:
                if "<" in rule:
                    c, v = rule.split("<")
                    v, dest = v.split(":")
                    print(dest)
                    if dest == "R":
                        hard_max_limits[c] = min(hard_max_limits[c], int(v) - 1)
                        continue
                    elif dest == "A":
                        soft_max_limits[c] = max(soft_max_limits[c] or int(v) - 1, int(v) - 1)
                        continue
                    curr_workflow = dest
                    break
                elif ">" in rule:
                    c, v = rule.split(">")
                    v, dest = v.split(":")
                    if dest == "R":
                        hard_min_limits[c] = max(hard_min_limits[c], int(v) + 1)
                        continue
                    elif dest == "A":
                        soft_min_limits[c] = min(soft_min_limits[c] or int(v) + 1, int(v) + 1)
                        continue
                    curr_workflow = dest
                    break
                else:
                    curr_workflow = rule
            if curr_workflow == "A":
                print("!!!!!!!!!!!!!!!!!!!!!")
                #breakpoint()
                break
            elif curr_workflow == "R":
                tmp = []
                print(list(reversed(soft_min_limits.items())))
                print(list(reversed(soft_max_limits.items())))
                #breakpoint()
                for l, u in zip(soft_min_limits.values(), soft_max_limits.values()):
                    if not l and not u:
                        continue
                    print(soft_min_limits, soft_min_limits)
                    l = l or 1
                    u = u or 4000
                    if u >= l:
                        tmp.append((u - l + 1))
                s += np.array(tmp).prod()
                break
            """
                tmp = []
                for l, u in zip(min_limits.values(), max_limits.values()):
                    if u >= l:
                        tmp.append((u - l + 1))
                print(tmp)
                s += np.array(tmp).prod()
            """
    return s


for example in puzzle.examples:
    example_answer = b(example.input_data)
    print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
    assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
