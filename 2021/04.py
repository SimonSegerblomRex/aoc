import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 4

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(input):
    input = input.split("\n\n")
    sequence, boards = input[0], input[1:]
    sequence = np.fromstring(sequence, dtype=int, sep=",")
    boards = np.dstack(
        np.fromstring(board.replace("\n", " "), dtype=int, sep=" ").reshape(5, 5)
        for board in boards
    )
    bingo = False
    state = np.full(boards.shape, False)
    for number in sequence:
        state |= boards == number
        for i in range(state.shape[2]):
            board_state = state[..., i]
            if board_state.all(axis=0).any() or board_state.all(axis=1).any():
                bingo = True
                break
        if bingo:
            return number * boards[~board_state, i].sum()

assert a(puzzle.example_data) == 4512
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(input):
    exit()

assert b(puzzle.example_data) == ...
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
