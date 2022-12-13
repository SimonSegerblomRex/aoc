import ast
import functools

from aocd.models import Puzzle

YEAR = 2022
DAY = 13

puzzle = Puzzle(year=YEAR, day=DAY)


def check(l0, l1):
    for v0, v1 in zip(l0, l1):
        if isinstance(v0, int) and isinstance(v1, int):
            if v0 < v1:
                return -1
            if v1 < v0:
                return 1
        else:
            if not isinstance(v0, list):
                v0 = [v0]
            if not isinstance(v1, list):
                v1 = [v1]
            if (ret := check(v0, v1)) != 0:
                return ret
    if len(l0) < len(l1):
        return -1
    if len(l1) < len(l0):
        return 1
    return 0


# Part a
def a(data):
    pairs = ([ast.literal_eval(line) for line in lines.split("\n")] for lines in data.split("\n\n"))
    return sum(i + 1 for i, p in enumerate(pairs) if check(*p) <= 0)


example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 13
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 6428


# Part b
def b(data):
    packets = [ast.literal_eval(line) for line in data.replace("\n\n", "\n").splitlines()]
    packets.extend([[[2]], [[6]]])
    packets = sorted(packets, key=functools.cmp_to_key(check))
    return (packets.index([[2]]) + 1) * (packets.index([[6]]) + 1)


example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 140
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 22464
