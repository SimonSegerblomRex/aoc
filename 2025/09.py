import datetime

import matplotlib.pyplot as plt
from shapely import Point, Polygon
from shapely.plotting import plot_polygon

from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

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
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == "50"
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    red = [tuple(map(int, line.split(","))) for line in data.splitlines()]
    red.append(red[0])

    polygon = Polygon(red)
    #plot_polygon(polygon)
    #plt.show()

    def is_green(x, y):
        return polygon.intersects(Point(x, y))

    m = 0
    for c0 in red:
        for c1 in red:
            tmp = area(*c0, *c1)
            if tmp <= m:
                continue
            #if not (is_green(c0[0], c1[1]) and is_green(c1[0], c0[1])):
            #    continue
            p2 = Polygon([c0, (c0[0], c1[1]), c1, (c1[0], c0[1]), c0])
            if not polygon.contains(p2):
                continue
            m = max(m, tmp)
    return m


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {24})")
        assert str(example_answer) == "24"
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
