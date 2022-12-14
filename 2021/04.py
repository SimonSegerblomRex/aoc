import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 4

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def _parse_bingo_data(data):
    data = data.split("\n\n")
    sequence, boards = data[0], data[1:]
    sequence = np.fromstring(sequence, dtype=int, sep=",")
    boards = np.dstack(
        [
            np.fromstring(board.replace("\n", " "), dtype=int, sep=" ").reshape(5, 5)
            for board in boards
        ]
    )
    return sequence, boards


def a(data):
    sequence, boards = _parse_bingo_data(data)
    state = np.full(boards.shape, False)
    for number in sequence:
        state |= boards == number
        for i in range(state.shape[2]):
            board_state = state[..., i]
            if board_state.all(axis=0).any() or board_state.all(axis=1).any():
                return number * boards[~board_state, i].sum()


assert a(puzzle.example_data) == 4512
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 34506


# Part b
def b(data):
    sequence, boards = _parse_bingo_data(data)
    bingo_scores = np.zeros(boards.shape[2], dtype=int)
    state = np.full(boards.shape, False)
    for number in sequence:
        state |= boards == number
        for i in np.flatnonzero(bingo_scores == 0):
            board_state = state[..., i]
            if board_state.all(axis=0).any() or board_state.all(axis=1).any():
                bingo_scores[i] = number * boards[~board_state, i].sum()
    return bingo_scores[i]


assert b(puzzle.example_data) == 1924
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 7686
