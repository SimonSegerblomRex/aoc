import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 17

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(x0, x1, y0, y1):
    """
    y_n = y + (y - 1) + (y - 2) + ... (y - n + 1)
        = n * y - n * (n - 1) / 2
        = (y + 1 / 2) * n - n**2 / 2

    Want to maximize y:

    0 = y + 1 / 2 - n
    => n = y + 1 / 2

    So with integer values we get

    n = y

    If the target zone is below 0 we know that the
    vertical speed will be -y when we are back at 0
    while falling down. Next step the speed and y
    position will be -(y + 1). To stay within area
    while maximizing y we should pick

    -(y + 1) = y0

    <=> y = -1 - y0
    """
    y_n = lambda n, y: n * (y - (n - 1) / 2)
    y = -1 - y0
    n = y
    return int(y_n(n, y))


example_answer = a(20, 30, -10, -5)
print(example_answer)
assert example_answer == 45
answer = a(281, 311, -74, -54)
assert answer == 2701
print("a:", answer)


# Part b
def b(x0, x1, y0, y1):
    # From part a we have:
    y_n = lambda n, y: n * (y - (n - 1) / 2)
    y_max = -1 - y0
    # y_min can't be smaller than y0
    y_min = y0
    # For x we have the limits
    x_max = x1
    x_min = 1
    # Check all combinations
    count = 0
    tmp2 = []
    for x_v in range(x_min, x_max + 1):
        for y_v in range(y_min, y_max + 1):
            x_p = 0
            y_p = 0
            x_vv = x_v
            y_vv = y_v
            while True:
                x_p += x_vv
                y_p += y_vv
                x_vv -= 1
                y_vv -= 1
                x_vv = max(x_vv, 0)
                if y_p < y0:
                    break
                if x_p > x1:
                    break
                if (x0 <= x_p <= x1) and (y0 <= y_p <= y1):
                    count += 1
                    break
    return count


example_answer = b(20, 30, -10, -5)
print(example_answer)
assert example_answer == 112
answer = b(281, 311, -74, -54)
print("b:", answer)
assert answer == 1070
