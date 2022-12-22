import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, debug=False):
    board, instructions = data.split("\n\n")
    board = board.replace(" ", "1")
    board = board.replace(".", "0")
    board = board.replace("#", "2")
    rows = [np.frombuffer(row.encode(), dtype=np.uint8) - ord("0") for row in board.splitlines()]
    width = max(map(len, rows))
    rows = [np.pad(row, (0, width - len(row)), constant_values=1) for row in rows]
    board = np.vstack(rows)
    board = np.pad(board, ((1, 1), (1, 1)), constant_values=1) #...
    height, width = board.shape
    instructions = re.findall("(\d+|[A-Z])", instructions)
    i = 1
    j = board[i, :].argmin()
    move_dir = (0, 1)
    debug_board = board.copy()
    while instructions:
        instruction = instructions.pop(0)
        if instruction.isdigit():
            for step in range(int(instruction)):
                next_i = (i + move_dir[0])
                next_j = (j + move_dir[1])
                if board[next_i, next_j] == 1:
                    if move_dir == (0, 1):
                        next_j = (board[i, :] != 1).argmax()
                    elif move_dir == (0, -1):
                        next_j = width - (board[i, :] != 1)[::-1].argmax() - 1
                    elif move_dir == (1, 0):
                        next_i = (board[:, j] != 1).argmax()
                    elif move_dir == (-1, 0):
                        next_i = height - (board[:, j] != 1)[::-1].argmax() - 1
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
        if debug:
            debug_symbol = {
                (0,1): 3,
                (1, 0): 4,
                (0, -1): 5,
                (-1, 0): 6,
            }
            debug_board[i, j] = debug_symbol[move_dir]
            #print(debug_board)
            #print(i, j, move_dir, instruction)
            #breakpoint()
    face_score = {
        (0,1): 0,
        (1, 0): 1,
        (0, -1): 2,
        (-1, 0): 3,
    }
    if debug:
        print(debug_board)
        print(i, j, move_dir)
    return int(i * 1000 + 4 * j + face_score[move_dir])
    breakpoint()

example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 6032
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 60362


# Part b
def b(data):
    exit()

example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == ...
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
