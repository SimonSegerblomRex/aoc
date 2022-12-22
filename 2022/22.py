import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    board, instructions = data.split("\n\n")
    board = board.replace(" ", "1")
    board = board.replace(".", "0")
    board = board.replace("#", "2")
    rows = [np.frombuffer(row.encode(), dtype=np.uint8) - ord("0") for row in board.splitlines()]
    width = max(map(len, rows))
    rows = [np.pad(row, (0, width - len(row)), constant_values=1) for row in rows]
    breakpoint()
    board = np.vstack(rows)
    height = board.shape[0]
    instructions = re.findall("(\d+|[A-Z])", instructions)
    i = 0
    j = board[i, :].argmin()
    move_dir = (0, 1)
    while instructions:
        instruction = instructions.pop(0)
        if instruction.isdigit():
            for step in range(int(instruction)):
                next_i = i + move_dir[0]
                next_j = j + move_dir[1]
                if board[next_i, next_j] == 1:
                    if move_dir == (0, 1):
                        next_j = board[i, :].argmin()
                    elif move_dir == (0, -1):
                        next_i = width - board[:, j][::-1].argmin()  # CHECK!!!
                    elif move_dir == (1, 0):
                        next_i = board[:, j].argmin()
                    elif move_dir == (-1, 0):
                        next_i = height - board[:, j][::-1].argmin()
                if board[next_i, next_j] == 2:
                    break
                i = next_i
                j = next_j
        else:
            if instruction == "L":
                if move_dir == (0, 1):
                    move_dir = (-1, 0)
                elif move_dir == (-1, 0):
                    move_dir = (0, -1)
                elif move_dir == (0, -1):
                    move_dir = (1, 0)
                elif move_dir == (1, 0):
                    move_dir = (0, 1)
            else:
                if move_dir == (0, 1):
                    move_dir = (1, 0)
                elif move_dir == (1, 0):
                    move_dir = (0, -1)
                elif move_dir == (0, -1):
                    move_dir = (-1, 0)
                elif move_dir == (-1, 0):
                    move_dir = (0, 1)
        if 0:
            board[i, j] = 3
            print(board)
            print(i, j, move_dir, instruction)
            breakpoint()
            board[i, j] = 0
    face_score = {
        (0,1): 0,
        (1, 0): 1,
        (0, -1): 2,
        (-1, 0): 3,
    }
    return (i - 1) * 1000 + 4 * (j - 1) + face_score[move_dir]
    breakpoint()

example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 6032
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    exit()

example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == ...
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
