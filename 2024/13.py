import re

from aocd.models import Puzzle

YEAR = 2024
DAY = 13

puzzle = Puzzle(year=YEAR, day=DAY)


pattern = r"""Button A: X\+(?P<dx_A>\d+), Y\+(?P<dx_Y>\d+)
Button B: X\+(?P<dx_B>\d+), Y\+(?P<dy_B>\d+)
Prize: X=(?P<goal_x>\d+), Y=(?P<goal_y>\d+)"""


# Part a
def a(data, extra=0):
    machines = [[int(n) for n in m] for m in re.findall(pattern, data)]
    s = 0
    for m in machines:
        x0, x1, x2, x3 = m[0], m[2], m[1], m[3]
        det = x0 * x3 - x1 * x2
        m[-1] += extra
        m[-2] += extra
        if det != 0:
            A = x3 * m[-2] - x1 * m[-1]
            B = -x2 * m[-2] + x0 * m[-1]
            if not (A % det) and not (B % det):
                s += (A // det) * 3 + B // det
    return s


answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 36954


# Part b
answer = a(puzzle.input_data, extra=10000000000000)
print("b:", answer)
assert answer == 79352015273424
