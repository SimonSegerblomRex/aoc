from collections import Counter, defaultdict
from functools import cache

from aocd.models import Puzzle

YEAR = 2024
DAY = 11

puzzle = Puzzle(year=YEAR, day=DAY)


def blinkhelper(stone):
    ss = str(stone)
    if stone == 0:
        return [1]
    elif not len(ss) % 2:
        w = len(ss)
        return [int(ss[:w//2]), int(ss[w//2:])]
    return [2024 * stone]


def blink(stones):
    out = defaultdict(int)
    for s in stones:
        new = blinkhelper(s)
        for n in new:
            out[n] += stones[s]
    return out


# Part a
def a(data, blinks=25):
    stones = list(map(int, data.split()))
    stones = Counter(stones)
    for _ in range(blinks):
        stones = blink(stones)
    return sum(Counter(stones).values())


example = "125 17"
answer = a(example)
print(answer)
assert answer == 55312
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
answer = a(puzzle.input_data, blinks=75)
print("b:", answer)
puzzle.answer_b = answer
