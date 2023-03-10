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
def b(data):
    exit()


example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == ...
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
