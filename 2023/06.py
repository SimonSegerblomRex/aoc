import numpy as np
from aocd.models import Puzzle

YEAR = 2023
DAY = 6

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(times, distances):
    number_of_ways = []
    for time_record, distance_record in zip(times, distances):
        distance_traveled = []
        for time_holding in range(1, time_record):
            distance_traveled.append((time_record - time_holding) * time_holding)
        distance_traveled = np.array(distance_traveled)
        number_of_ways.append(len(distance_traveled[distance_traveled > distance_record]))
    return np.prod(number_of_ways)


example_answer = a(times=[7, 15, 30], distances=[9, 40, 200])
assert example_answer == 288
answer = a(times=[51, 69, 98, 78], distances=[377, 1171, 1224, 1505])
print("a:", answer)
assert answer == 131376


# Part b
example_answer = a(times=[71530], distances=[940200])
assert example_answer == 71503
answer = a(times=[51699878], distances=[377117112241505])
print("b:", answer)
assert answer == 34123437
