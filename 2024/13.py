import datetime
import re
from scipy.linalg import inv

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


pattern = r"""Button A: X\+(?P<dx_A>\d+), Y\+(?P<dx_Y>\d+)
Button B: X\+(?P<dx_B>\d+), Y\+(?P<dy_B>\d+)
Prize: X=(?P<goal_x>\d+), Y=(?P<goal_y>\d+)"""


# Part a
def a(data):
    machines = []
    machines = re.findall(pattern, data)
    machines = [[int(n) for n in m] for m in machines]
    s = 0
    for m in machines:
        #tmp = np.linalg.inv(np.array([[m[0], m[2]], [m[1], m[3]]], dtype=int).reshape(2, 2))
        x0, x1, x2, x3 =  m[0], m[2], m[1], m[3]
        det = x0*x3 - x1*x2
        if det != 0:
            A = (x3*m[-2] - x1*m[-1])
            B = (-x2*m[-2] + x0*m[-1])
            if not (A % det) and not (B % det):
                s += (A // det) * 3 + B // det
    return s


answer = a(puzzle.examples[0].input_data)
print(answer)
assert answer == 480
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer
breakpoint()


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
