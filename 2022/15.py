import re
from collections import Counter

from aocd.models import Puzzle

YEAR = 2022
DAY = 15

puzzle = Puzzle(year=YEAR, day=DAY)

PATTERN = (
    r"Sensor at x=([-\d]+), y=([-\d]+): closest beacon is at x=([-\d]+), y=([-\d]+)"
)

# Part a
def a(data, y):
    data = re.findall(PATTERN, data)
    data = [tuple(map(int, e)) for e in data]
    x = set()
    exclude = set()
    for xs, ys, xb, yb in data:
        md = abs(xs - xb) + abs(ys - yb)
        if (yd := abs(ys - y)) <= md:
            x.update(range(xs - (md - yd), xs + (md - yd) + 1))
        if yb == y:
            exclude.add(xb)

    return len(x - exclude)


example_answer = a(puzzle.example_data, y=10)
print(example_answer)
assert example_answer == 26
answer = a(puzzle.input_data, y=2000000)
print("a:", answer)
assert answer == 5367037


# Part b
def find_x(data, y, max_c):
    x = set()
    for xs, ys, xb, yb in data:
        md = abs(xs - xb) + abs(ys - yb)
        if (yd := abs(ys - y)) <= md:
            x.update(range(xs - (md - yd), xs + (md - yd) + 1))
    x = set(range(max_c + 1)) - x

    return x


def b(data, max_c):
    data = re.findall(PATTERN, data)
    data = [tuple(map(int, e)) for e in data]
    lines = []
    for xs, ys, xb, yb in data:
        md = abs(xs - xb) + abs(ys - yb)
        # [(x0, x1), (k, m)] (y = k*x + m)
        lines.append([(xs - md - 1, xs), (1, ys - (xs - md - 1))])
        lines.append([(xs, xs + md + 1), (1, ys - md - 1 - xs)])
        lines.append([(xs - md - 1, xs), (-1, ys + xs - md - 1)])
        lines.append([(xs, xs + md + 1), (-1, ys + md + 1 + xs)])
    yy = []
    for i, curr_line in enumerate(lines):
        for j, other_line in enumerate(lines):
            if j == i:
                continue
            if curr_line[1][0] == other_line[1][0]:
                continue
            x0 = max(curr_line[0][0], other_line[0][0])
            x1 = min(curr_line[0][1], other_line[0][1])
            x = (other_line[1][1] - curr_line[1][1]) // 2
            y = curr_line[1][0] * x + curr_line[1][1]
            if (x0 <= x <= x1) and (0 <= x <= max_c) and (0 <= y <= max_c):
                yy.append(y)

    cc = Counter(yy)
    for y, _ in cc.most_common():
        x = find_x(data, y, max_c)
        if x:
            break
    return 4000000 * list(x)[0] + y


example_answer = b(puzzle.example_data, max_c=20)
print(example_answer)
assert example_answer == 56000011
answer = b(puzzle.input_data, max_c=4000000)
print("b:", answer)
assert answer == 11914583249288
