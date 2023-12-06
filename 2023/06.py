import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(times, distances):
    ttt = []
    for t, d in zip(times, distances):
        tt = []
        for i in range(1, t + 1):
            tt.append((t - i) * i)
        tt = np.array(tt)
        #breakpoint()
        ttt.append(len(tt[tt > d])) #>=?
    return np.prod(ttt)
    breakpoint()


example_answer = a(times=[7, 15, 30], distances=[9, 40, 200])
#print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
#assert str(example_answer) == example.answer_a
answer = a(times=[51  ,   69    , 98   ,  78], distances = [377  , 1171  , 1224 ,  1505])
print("a:", answer)
#puzzle.answer_a = answer


# Part b
def b(times, distances):
    ttt = []
    for t, d in zip(times, distances):
        tt = []
        for i in range(1, t + 1):
            tt.append((t - i) * i)
        tt = np.array(tt)
        ttt.append(len(tt[tt > d]))
    return np.prod(ttt)
    breakpoint()


example_answer = b(times=[71530], distances=[940200])
#print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
assert example_answer == 71503
answer = b(times=[51699878], distances = [377117112241505])
print("b:", answer)
puzzle.answer_b = answer
