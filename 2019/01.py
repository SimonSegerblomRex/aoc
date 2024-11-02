from aocd.models import Puzzle

YEAR = 2019
DAY = 1

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    s = 0
    for m in map(int, data.splitlines()):
        s += m // 3 - 2
    return s


answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 3358992


# Part b
def b(data):
    s = 0
    for m in map(int, data.splitlines()):
        while True:
            f = m // 3 - 2
            if f <= 0:
                break
            s += f
            m = f
    return s


answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 5035632
