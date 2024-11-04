from functools import cache
from operator import itemgetter

import numpy as np
from sympy import Abs, arg, I, im, pi, re
from aocd.models import Puzzle

YEAR = 2019
DAY = 10

puzzle = Puzzle(year=YEAR, day=DAY)

@cache
def hmm(x):
    tmp = x / Abs(x)
    # sympy doesn't simplify the fractions
    # if we don't do anything special...
    #return tmp.ratsimp()
    #return re(tmp), im(tmp)
    #return tmp.nsimplify()
    #return tmp.powsimp()
    #return arg(tmp)
    return tmp.expand()


# Part a
def a(data):
    asteroids = []
    for i, line in enumerate(data.splitlines()):
        for j, c in enumerate(line):
            if c == "#":
                asteroids.append(j + i*I)
    best_score = 0
    best_coords = None
    for a in asteroids:
        tmp = set()
        for b in asteroids:
            if a == b:
                continue
            tmp.add(hmm(b - a))
        score = len(tmp)
        if score > best_score:
            best_score = score
            best_coords = a
    print(best_coords)
    return best_score


if 0:
    for example in puzzle.examples:
        if example.answer_a:
            example_answer = a(example.input_data)
            print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
            assert str(example_answer) == example.answer_a
    answer = a(puzzle.input_data)
    print("a:", answer)
    assert answer == 309


# Part b
def b(data, start=37 + 25*I):
    # best coord (print from a): 37 + 25*I
    asteroids = []
    for i, line in enumerate(data.splitlines()):
        for j, c in enumerate(line):
            if c == "#":
                asteroids.append(j + i*I)
    a = start
    asteroids.remove(a)
    # (angle, abs, coord)
    polar = (((pi / 2 + arg(b - a)) % (2 * pi), abs(b - a), b) for b in asteroids)
    polar = sorted(polar)
    pos = 0
    prev_angle = -1
    count = 0
    while count < 200:
        if polar[pos][0] == prev_angle:
            pos += 1
        elif polar[pos][0] > prev_angle:
            tmp = polar.pop(pos)
            prev_angle = tmp[0]
            count += 1
            if count in [1, 2, 3, 10, 20, 50, 100, 199, 200, 201, 299]:
                print(count, re(tmp[2]), im(tmp[2]))
        else:
            # FIXME
            pos = 0
            breakpoint()
    return int(re(tmp[2]) * 100 +  im(tmp[2]))


answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 406
