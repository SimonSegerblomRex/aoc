import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)

from functools import cache


# Part a
def a(data):
    devices = {}
    for line in data.splitlines():
        key, val = line.split(":")
        devices[key] = val.strip().split(" ")
    @cache
    def find_path(start, goal):
        if start == goal:
            return 1
        c = 0
        for device in devices[start]:
            c +=find_path(device, goal)
        return c
    return find_path("you", "out")
    breakpoint()


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == "5"
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    breakpoint()
    devices = {}
    for line in data.splitlines():
        key, val = line.split(":")
        devices[key] = val.strip().split(" ")
    @cache
    def find_path(start, goal, goal0=False, goal1=False):
        if start == goal:
            if goal0 and goal1:
                return 1
            return 0
        c = 0
        for device in devices[start]:
            c += find_path(device, goal, goal0=goal0 or device == "fft", goal1=goal1 or device == "dac")
        return c
    return find_path("svr", "out")


example_answer = b("""svr: aaa bbb
aaa: fft
fft: ccc
bbb: tty
tty: ccc
ccc: ddd eee
ddd: hub
hub: fff
eee: dac
dac: fff
fff: ggg hhh
ggg: out
hhh: out
""")
print(f"Example answer: {example_answer} (expecting: 2)")
assert str(example_answer) == "2"
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
