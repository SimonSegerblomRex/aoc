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


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {50})")
        assert str(example_answer) == "50"
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 4769758290


# Part b
def b(data):
    red = [tuple(map(int, line.split(","))) for line in data.splitlines()]
    vertices = set(red)
    red.append(red[0])
    lines = set(zip(red, red[1:]))

    m = 0
    for c0 in red:
        for c1 in red:
            candidate_area = area(*c0, *c1)
            if candidate_area <= m:
                continue
            x_min, x_max = sorted([c0[0], c1[0]])
            y_min, y_max = sorted([c0[1], c1[1]])
            x_lim = range(x_min + 1, x_max)
            y_lim = range(y_min + 1, y_max)
            if not x_lim or not y_lim:
                continue
            # Check so that no vertices are inside the rectangle
            for x, y in vertices:
                if (x in x_lim) and (y in y_lim):
                    break
            else:
                # Check lines
                for v0, v1 in lines:
                    lx_min, lx_max = sorted([v0[0], v1[0]])
                    ly_min, ly_max = sorted([v0[1], v1[1]])
                    if ly_min == ly_max:
                        if lx_min <= x_min and lx_max >= x_max and y_min < ly_min < y_max:
                            break
                    elif lx_min == lx_max:
                        if ly_min <= y_min and ly_max >= y_max and x_min < lx_min < x_max:
                            break
                    else:
                        raise RuntimeError("shouldn't end up here...")
                else:
                    m = max(m, candidate_area)
    return m


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {24})")
        #assert str(example_answer) == "24"
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 1588990708
