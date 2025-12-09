from aocd.models import Puzzle

YEAR = 2025
DAY = 9

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def area(x0, y0, x1, y1):
    return (abs(x1 - x0) + 1) * (abs(y1 - y0) + 1)


def a(data):
    coords = [tuple(map(int, line.split(","))) for line in data.splitlines()]
    m = 0
    for c0 in coords:
        for c1 in coords:
            m = max(m, area(*c0, *c1))
    return m


assert (answer := a(puzzle.examples[0].input_data)) == 50, answer
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 4769758290


# Part b
def b(data):
    red = [tuple(map(int, line.split(","))) for line in data.splitlines()]
    vertices = set(red)
    red.append(red[0])
    lines = zip(red, red[1:])
    vertical = set()
    horizontal = set()
    for (x0, y0), (x1, y1) in lines:
        if x0 == x1:
            vertical.add((x0, min(y0, y1), max(y0, y1)))
        else:
            horizontal.add((y0, min(x0, x1), max(x0, x1)))

    m = 0
    for c0 in vertices:
        for c1 in vertices:
            if c0 == c1:
                continue
            candidate_area = area(*c0, *c1)
            if candidate_area <= m:
                continue
            x_min, x_max = sorted([c0[0], c1[0]])
            y_min, y_max = sorted([c0[1], c1[1]])
            for x, y0, y1 in vertical:
                if y0 < y_max and y1 > y_min and x_min < x < x_max:
                    break
            else:
                for y, x0, x1 in horizontal:
                    if x0 < x_max and x1 > x_min and y_min < y < y_max:
                        break
                else:
                    m = max(m, candidate_area)
    return m


assert (answer := b(puzzle.examples[0].input_data)) == 24, answer
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 1588990708
