import datetime
import re
from itertools import groupby

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)

SCORES_HANDS = {
    "five": 36,
    "four": 35,
    "full": 34,
    "three": 33,
    "two-pair": 32,
    "one-pair": 31,
    "high": 30,
}

SCORES_CARDS = {
    "A": 22,
    "K": 21,
    "Q": 20,
    "J": 19,
    "T": 18,
    "9": 17,
    "8": 16,
    "7": 15,
    "6": 14,
    "5": 13,
    "4": 12,
    "3": 11,
    "2": 10,
}



def evaluate(hand):
    counts = np.array([hand.count(c) for c in SCORES_CARDS])
    if counts.max() == 5:
        return int("".join(str(s) for s in [SCORES_HANDS["five"], *(SCORES_CARDS[c] for c in hand)]))
    if counts.max() == 4:
        return int("".join(str(s) for s in [SCORES_HANDS["four"], *(SCORES_CARDS[c] for c in hand)]))
    if 0:
        g = groupby(counts, key=lambda c: c > 0)
        g = [len(list(s)) for v, s in g if v]
        if max(g) == 5:
            return SCORES_HANDS["..."] + SCORES_CARDS[hand[0]]
    if counts.max() == 3:
        if sorted(counts)[-2] == 2:
            return int("".join(str(s) for s in [SCORES_HANDS["full"], *(SCORES_CARDS[c] for c in hand)]))
        return int("".join(str(s) for s in [SCORES_HANDS["three"], *(SCORES_CARDS[c] for c in hand)]))
    if counts.max() == 2:
        if sorted(counts)[-2] == 2:
            return int("".join(str(s) for s in [SCORES_HANDS["two-pair"], *(SCORES_CARDS[c] for c in hand)]))
        return int("".join(str(s) for s in [SCORES_HANDS["one-pair"], *(SCORES_CARDS[c] for c in hand)]))
    return int("".join(str(s) for s in [SCORES_HANDS["high"], *(SCORES_CARDS[c] for c in hand)]))


# Part a
def a(data):
    hands = []
    for line in data.splitlines():
        hand, bid = line.split(" ")
        hands.append([hand, int(bid)])
    for hand in hands:
        hand.append(evaluate(hand[0]))
    hands = sorted(hands, key=lambda e: e[2])
    s = 0
    for rank, hand in enumerate(hands, 1):
        s += rank * hand[1]
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print(answer)
assert answer > 246758041
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    breakpoint()


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
