import ast

import numpy as np
from aocd.models import Puzzle

YEAR = 2023
DAY = 24

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, lim0, lim1):
    positions = []
    velocities = []
    for line in data.splitlines():
        pos, v = line.split("@")
        pos = ast.literal_eval(pos)
        v = ast.literal_eval(v.strip())
        positions.extend(list(pos))
        velocities.extend(list(v))
    positions = np.array(positions, dtype=np.int64)
    positions.shape = (-1, 3)
    velocities = np.array(velocities, dtype=np.int64)
    velocities.shape = (-1, 3)

    positions = positions[:, :2]
    velocities = velocities[:, :2]

    positions -= lim0
    lim1 -= lim0
    lim0 -= lim0

    positions = positions.astype(float)
    positions /= lim1
    lim1 /= lim1

    s = 0
    for p, v in zip(positions, velocities):
        p1 = p
        p2 = p + v
        p3 = positions
        p4 = positions + velocities
        x = (
            (p1[0] * p2[1] - p1[1] * p2[0]) * (p3[:, 0] - p4[:, 0])
            - (p1[0] - p2[0]) * (p3[:, 0] * p4[:, 1] - p3[:, 1] * p4[:, 0])
        ) / (
            (p1[0] - p2[0]) * (p3[:, 1] - p4[:, 1])
            - (p1[1] - p2[1]) * (p3[:, 0] - p4[:, 0])
        )
        y = (
            (p1[0] * p2[1] - p1[1] * p2[0]) * (p3[:, 1] - p4[:, 1])
            - (p1[1] - p2[1]) * (p3[:, 0] * p4[:, 1] - p3[:, 1] * p4[:, 0])
        ) / (
            (p1[0] - p2[0]) * (p3[:, 1] - p4[:, 1])
            - (p1[1] - p2[1]) * (p3[:, 0] - p4[:, 0])
        )
        tmp = (x >= lim0) & (x <= lim1) & (y >= lim0) & (y <= lim1)
        t1 = (x - p[0]) / v[0]
        t2 = (y - p[1]) / v[1]
        t3 = (x - p3[:, 0]) / velocities[:, 0]
        t4 = (y - p3[:, 1]) / velocities[:, 1]
        # breakpoint()
        s += sum(tmp & (t1 > 0) & (t2 > 0) & (t3 > 0) & (t4 > 0))
    return s // 2


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data, 7, 27)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert example_answer == 2
answer = a(puzzle.input_data, 200000000000000, 400000000000000)
print("a:", answer)
assert answer == 12783

# Part b
def b(data):
    positions = []
    velocities = []
    for line in data.splitlines():
        pos, v = line.split("@")
        pos = ast.literal_eval(pos)
        v = ast.literal_eval(v.strip())
        positions.extend(list(pos))
        velocities.extend(list(v))
    positions = np.array(positions, dtype=int)
    positions.shape = (-1, 3)
    velocities = np.array(velocities, dtype=int)
    velocities.shape = (-1, 3)

    from sympy import solve
    from sympy import symbols
    t_symbols = symbols(f"t:{5}", integer=True, positive=True)
    s_x, s_y, s_z = symbols("s_x,s_y,s_z", integer=True)
    v_x, v_y, v_z = symbols("v_x,v_y,v_z", integer=True)
    equations = []
    i = 0
    for t, p, v in zip(t_symbols, positions, velocities):
        equations.append(s_x + t * v_x - p[0] - t * v[0])
        equations.append(s_y + t * v_y - p[1] - t * v[1])
        equations.append(s_z + t * v_z - p[2] - t * v[2])
        i += 1
        if i > 4:
            break
    solution = solve(equations, [s_x, s_y, s_z, v_x, v_y, v_z, *t_symbols], dict=True)
    breakpoint()
    return solution[0][s_x] + solution[0][s_y] + solution[0][s_z]


example_answer = b(example.input_data)
print(f"Example answer: {example_answer} (expecting: {47})")
assert example_answer == 47
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
