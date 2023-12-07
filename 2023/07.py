from itertools import groupby

import numpy as np
from aocd.models import Puzzle

YEAR = 2023
DAY = 7

puzzle = Puzzle(year=YEAR, day=DAY)

SCORES_HANDS = {
    (5, 0): 36,
    (4, 1): 35,
    (3, 2): 34,
    (3, 1): 33,
    (2, 2): 32,
    (2, 1): 31,
    (1, 1): 30,
}

SCORES_CARDS = {
    "A": 23,
    "K": 22,
    "Q": 21,
    "J": 20,
    "T": 19,
    "9": 18,
    "8": 17,
    "7": 16,
    "6": 15,
    "5": 14,
    "4": 13,
    "3": 12,
    "2": 11,
}


def evaluate(hand):
    counts = [hand.count(c) for c in SCORES_CARDS]
    counts = sorted(counts, reverse=True)
    main_score = SCORES_HANDS[tuple(counts[:2])]
    card_score = (SCORES_CARDS[c] for c in hand)
    return int("".join(str(s) for s in [main_score, *card_score]))


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
print("a:", answer)
assert answer == 246912307


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
