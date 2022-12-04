from pathlib import Path

from aocd import get_data, submit

puzzle = Puzzle(year=2015, day=1)

data = puzzle.input_data

examples = [
    ("2x3x4", 58),
    ("1x1x10", 43),
]


def solve(data):
    # 2*6 + 2*12 + 2*8 = 52 square feet of wrapping paper plus 6
    l, w, h = [int(x) for x in data.split("x")]
    s1 = l * w
    s2 = w * h
    s3 = h * l
    return 2 * s1 + 2 * s2 + 2 * s3 + min((s1, s2, s3))


for data, answer in examples:
    assert solve(data) == answer

total = sum([solve(data) for data in data.split("\n")])
print("a:", total)

# part b
examples = [
    ("2x3x4", 34),
    ("1x1x10", 14),
]


def solve_b(data):
    l, w, h = [int(x) for x in data.split("x")]
    l1, l2 = sorted([l, w, h])[:2]
    volume = l * w * h
    return 2 * l1 + 2 * l2 + volume


for data, answer in examples:
    assert solve_b(data) == answer

total = sum([solve_b(data) for data in data.split("\n")])
print("b:", total)
