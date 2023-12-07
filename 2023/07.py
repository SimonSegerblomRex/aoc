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
SCORES_CARDS["J"] = 10

SCORES_HANDS = {
    # Five of a kind
    (5, 0, 0): 36,
    (4, 0, 1): 36,
    (3, 0, 2): 36,
    (2, 0, 3): 36,
    (1, 0, 4): 36,
    (0, 0, 5): 36,
    # Four of a kind
    (4, 1, 0): 35,
    (3, 1, 1): 35,
    (2, 1, 2): 35,
    (1, 1, 3): 35,
    # Full house
    (3, 2, 0): 34,
    (2, 2, 1): 34,
    # Three of a kind
    (3, 1, 0): 33,
    (2, 1, 1): 33,
    (1, 1, 2): 33,
    # Two pairs
    (2, 2, 0): 32,
    # One pair
    (2, 1, 0): 31,
    (1, 1, 1): 31,
    # High card
    (1, 1, 0): 30,
}


def evaluate(hand):
    counts = [hand.count(c) for c in SCORES_CARDS]
    nbr_jokers = counts.pop(3)
    counts = sorted(counts, reverse=True)
    main_score = SCORES_HANDS[(*counts[:2], nbr_jokers)]
    card_score = (SCORES_CARDS[c] for c in hand)
    return int("".join(str(s) for s in [main_score, *card_score]))


for example in puzzle.examples:
    if example.answer_b:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = a(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
