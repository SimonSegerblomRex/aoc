import datetime

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


O_ROCK = "A"
O_PAPER = "B"
O_SCISSORS = "C"

Y_ROCK = "X"
Y_PAPER = "Y"
Y_SCISSORS = "Z"

S_ROCK = 1
S_PAPER = 2
S_SCISSORS = 3

S_LOSS = 0
S_DRAW = 3
S_WIN = 6

SCORES = {
    (O_ROCK, Y_ROCK): S_ROCK + S_DRAW,
    (O_ROCK, Y_PAPER): S_PAPER + S_WIN,
    (O_ROCK, Y_SCISSORS): S_SCISSORS + S_LOSS,
    (O_PAPER, Y_ROCK): S_ROCK + S_LOSS,
    (O_PAPER, Y_PAPER): S_PAPER + S_DRAW,
    (O_PAPER, Y_SCISSORS): S_SCISSORS + S_WIN,
    (O_SCISSORS, Y_ROCK): S_ROCK + S_WIN,
    (O_SCISSORS, Y_PAPER): S_PAPER + S_LOSS,
    (O_SCISSORS, Y_SCISSORS): S_SCISSORS + S_DRAW,
}

# Part a
def a(input):
    input = [i.split(" ") for i in input.split("\n")]
    score = 0
    for round in input:
        score +=  SCORES[tuple(round)]
    print(score)
    return score

#assert a(puzzle.example_data) == ...
answer = a(puzzle.input_data)
print("a:", answer)
#puzzle.answer_a = answer


# Part b
O_ROCK = "A"
O_PAPER = "B"
O_SCISSORS = "C"

Y_LOSS = "X"
Y_DRAW = "Y"
Y_WIN = "Z"

S_ROCK = 1
S_PAPER = 2
S_SCISSORS = 3

S_LOSS = 0
S_DRAW = 3
S_WIN = 6

SCORES = {
    (O_ROCK, Y_LOSS): S_SCISSORS + S_LOSS,
    (O_ROCK, Y_DRAW): S_ROCK + S_DRAW,
    (O_ROCK, Y_WIN): S_PAPER + S_WIN,
    (O_PAPER, Y_LOSS): S_ROCK + S_LOSS,
    (O_PAPER, Y_DRAW): S_PAPER + S_DRAW,
    (O_PAPER, Y_WIN): S_SCISSORS + S_WIN,
    (O_SCISSORS, Y_LOSS): S_PAPER + S_LOSS,
    (O_SCISSORS, Y_DRAW): S_SCISSORS + S_DRAW,
    (O_SCISSORS, Y_WIN): S_ROCK + S_WIN,
}

assert a(puzzle.example_data) == 12
answer = a(puzzle.input_data)
print("b:", answer)
#puzzle.answer_b = answer
