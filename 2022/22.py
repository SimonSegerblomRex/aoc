import datetime
import re

import matplotlib.pyplot as plt
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
    rows = [
        np.frombuffer(row.encode(), dtype=np.uint8) - ord("0")
        for row in board.splitlines()
    ]
    width = max(map(len, rows))
    rows = [np.pad(row, (0, width - len(row)), constant_values=1) for row in rows]
    board = np.vstack(rows)
    board = np.pad(board, ((1, 1), (1, 1)), constant_values=1)  # ...
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
                next_i = i + move_dir[0]
                next_j = j + move_dir[1]
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
                (0, 1): 3,
                (1, 0): 4,
                (0, -1): 5,
                (-1, 0): 6,
            }
            debug_board[i, j] = debug_symbol[move_dir]
            # print(debug_board)
            # print(i, j, move_dir, instruction)
            # breakpoint()
    face_score = {
        (0, 1): 0,
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
def b(data, debug=False, example=False):
    board, instructions = data.split("\n\n")
    board = board.replace(" ", "1")
    board = board.replace(".", "0")
    board = board.replace("#", "2")
    rows = [
        np.frombuffer(row.encode(), dtype=np.uint8) - ord("0")
        for row in board.splitlines()
    ]
    width = max(map(len, rows))
    rows = [np.pad(row, (0, width - len(row)), constant_values=1) for row in rows]
    board = np.vstack(rows)
    height, width = board.shape
    side = min(height, width) // 3
    tmp = board.reshape(height // side, side, width // side, side)
    tmp = tmp.swapaxes(1, 2)
    faces = {}
    if example:
        # Example data:
        faces[1] = tmp[0, 2]
        faces[2] = np.rot90(tmp[2, 3], 2)
        faces[3] = np.rot90(tmp[2, 2], 2)
        faces[4] = np.rot90(tmp[1, 1], -1)
        faces[5] = np.rot90(tmp[1, 0], 2)
        faces[6] = tmp[1, 2]
        # Dummy
        faces[0] = tmp[0, 0]
    else:
        # Input data:
        faces[1] = tmp[0, 1]
        faces[2] = tmp[0, 2]
        faces[3] = np.rot90(tmp[2, 1], 2)
        faces[4] = np.rot90(tmp[2, 0], 2)
        faces[5] = np.rot90(tmp[3, 0], 1)
        faces[6] = tmp[1, 1]
        # Dummy
        faces[0] = tmp[0, 0]

    if example:
        test = np.block(
            [
                [faces[0], faces[0], faces[1], faces[0]],
                [np.rot90(faces[5], 2), np.rot90(faces[4], 1), faces[6], faces[0]],
                [faces[0], faces[0], np.rot90(faces[3], 2), np.rot90(faces[2], 2)],
            ]
        )
    else:
        test = np.block(
            [
                [faces[0], faces[1], faces[2]],
                [faces[0], faces[6], faces[0]],
                [np.rot90(faces[4], 2), np.rot90(faces[3], 2), faces[0]],
                [np.rot90(faces[5], -1), faces[0], faces[0]],
            ]
        )
    np.testing.assert_array_equal(board, test)

    instructions = re.findall("(\d+|[A-Z])", instructions)
    next_face_map = {
        (0, 1): {
            1: 2,
            2: 3,
            3: 4,
            4: 1,
            5: 2,
            6: 2,
        },
        (0, -1): {
            1: 4,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 4,
        },
        (1, 0): {
            1: 6,
            2: 6,
            3: 6,
            4: 6,
            5: 1,
            6: 3,
        },
        (-1, 0): {
            1: 5,
            2: 5,
            3: 5,
            4: 5,
            5: 3,
            6: 1,
        },
    }
    curr_face = 1
    i = 0
    j = faces[curr_face].argmin()
    move_dir = (0, 1)
    debug_board = board.copy()
    debug_set = set()
    debug_next = []
    plt.figure(figsize=(6, 9), frameon=False, layout="tight")
    debug_counter = 0
    while instructions:
        debug_counter += 1
        instruction = instructions.pop(0)
        if instruction.isdigit():
            for step in range(int(instruction)):
                next_face = curr_face
                next_move_dir = move_dir
                next_i = i + move_dir[0]
                next_j = j + move_dir[1]
                if next_j >= side:
                    assert move_dir == (0, 1)
                    next_face = next_face_map[move_dir][curr_face]
                    if curr_face == 5:
                        next_i = 0
                        next_j = side - 1 - i
                        next_move_dir = (1, 0)
                        debug_set.add("A")
                    elif curr_face == 6:
                        next_i = side - 1
                        next_j = i
                        next_move_dir = (-1, 0)
                        debug_set.add("B")
                    else:
                        next_j = 0
                        debug_set.add("C")
                elif next_j < 0:
                    assert move_dir == (0, -1)
                    next_face = next_face_map[move_dir][curr_face]
                    if curr_face == 5:
                        next_i = 0
                        next_j = i
                        next_move_dir = (1, 0)
                        debug_set.add("D")
                    elif curr_face == 6:
                        next_i = side - 1
                        next_j = side - 1 - i
                        next_move_dir = (-1, 0)
                        debug_set.add("E")
                    else:
                        next_j = side - 1
                        debug_set.add("F")
                elif next_i >= side:
                    assert move_dir == (1, 0)
                    next_face = next_face_map[move_dir][curr_face]
                    if curr_face == 2:
                        next_i = j
                        next_j = side - 1
                        next_move_dir = (0, -1)
                        debug_set.add("G")
                    elif curr_face == 3:
                        next_i = side - 1
                        next_j = side - 1 - j
                        next_move_dir = (-1, 0)
                        debug_set.add("H")
                    elif curr_face == 4:
                        next_i = side - 1 - j
                        next_j = 0
                        next_move_dir = (0, 1)
                        debug_set.add("I")
                    elif curr_face == 6:
                        next_i = side - 1
                        next_j = side - 1 - j
                        next_move_dir = (-1, 0)
                        debug_set.add("J")
                    else:
                        next_i = 0
                        debug_set.add("K")
                elif next_i < 0:
                    assert move_dir == (-1, 0)
                    next_face = next_face_map[move_dir][curr_face]
                    if curr_face == 2:
                        next_i = side - 1 - j
                        next_j = side - 1
                        next_move_dir = (0, -1)
                        debug_set.add("L")
                    elif curr_face == 3:
                        next_i = 0
                        next_j = side - 1 - j
                        next_move_dir = (1, 0)
                        debug_set.add("M")
                    elif curr_face == 4:
                        next_i = j
                        next_j = 0
                        next_move_dir = (0, 1)
                        debug_set.add("N")
                    elif curr_face == 5:
                        next_i = 0
                        next_j = side - 1 - j
                        next_move_dir = (1, 0)
                        debug_set.add("O")
                    else:
                        next_i = side - 1
                        debug_set.add("P")

                if 0:
                    if example:
                        # For example data
                        faces[curr_face][i, j] = 4
                        tmp2 = np.block(
                            [
                                [faces[0], faces[0], faces[1], faces[0]],
                                [
                                    np.rot90(faces[5], 2),
                                    np.rot90(faces[4], 1),
                                    faces[6],
                                    faces[0],
                                ],
                                [
                                    faces[0],
                                    faces[0],
                                    np.rot90(faces[3], 2),
                                    np.rot90(faces[2], 2),
                                ],
                            ]
                        )
                        faces[curr_face][i, j] = 0
                    else:
                        faces[curr_face][i, j] = 4
                        tmp2 = np.block(
                            [
                                [faces[0], faces[1], faces[2]],
                                [faces[0], faces[6], faces[0]],
                                [
                                    np.rot90(faces[4], 2),
                                    np.rot90(faces[3], 2),
                                    faces[0],
                                ],
                                [np.rot90(faces[5], -1), faces[0], faces[0]],
                            ]
                        )
                        faces[curr_face][i, j] = 0
                    # print(tmp2)
                    if 0:
                        if curr_face != next_face:
                            debug_next = [1, 1]
                        if debug_next:
                            debug_next.pop()
                            plt.imshow(tmp2, aspect="equal")  # ;plt.show()
                            plt.draw()
                            print(curr_face)
                            plt.pause(0.01)
                            plt.clf()
                            breakpoint()
                    elif 1:
                        print(
                            "curr_face:",
                            curr_face,
                            "i:",
                            i,
                            "j:",
                            j,
                            "move_dir:",
                            move_dir,
                            "instruction:",
                            debug_counter,
                        )
                        plt.imshow(tmp2, aspect="equal")  # ;plt.show()
                        plt.draw()
                        plt.pause(0.05)
                        plt.clf()
                    # breakpoint()

                if faces[next_face][next_i, next_j] == 2:
                    break
                curr_face = next_face
                move_dir = next_move_dir
                i = next_i
                j = next_j

        else:
            """
            move_dir_orig = move_dir
            if instruction == "L":
                move_dir2 = tuple(np.array([[0, -1], [1, 0]]) @ move_dir)
            else:
                move_dir2 = tuple(np.array([[0, 1], [-1, 0]]) @ move_dir)
            """
            if instruction == "L":
                if move_dir == (0, 1):
                    move_dir = (-1, 0)
                elif move_dir == (-1, 0):
                    move_dir = (0, -1)
                elif move_dir == (0, -1):
                    move_dir = (1, 0)
                elif move_dir == (1, 0):
                    move_dir = (0, 1)
            elif instruction == "R":
                if move_dir == (0, 1):
                    move_dir = (1, 0)
                elif move_dir == (1, 0):
                    move_dir = (0, -1)
                elif move_dir == (0, -1):
                    move_dir = (-1, 0)
                elif move_dir == (-1, 0):
                    move_dir = (0, 1)
            else:
                raise ValueError(f"Unknown instruction '{instruction}'")
            """
            assert move_dir == move_dir2
            """
        if debug:
            debug_symbol = {
                (0, 1): 3,
                (1, 0): 4,
                (0, -1): 5,
                (-1, 0): 6,
            }
            debug_board[i, j] = debug_symbol[move_dir]
            # print(debug_board)
            # print(i, j, move_dir, instruction)
            # breakpoint()
    face_score = {
        (0, 1): 0,
        (1, 0): 1,
        (0, -1): 2,
        (-1, 0): 3,
    }

    if debug:
        print(debug_board)
    print(i, j, move_dir, curr_face)
    if example:
        # For example data
        faces[curr_face][i, j] = 4
        tmp2 = np.block(
            [
                [faces[0], faces[0], faces[1], faces[0]],
                [np.rot90(faces[5], 2), np.rot90(faces[4], 1), faces[6], faces[0]],
                [faces[0], faces[0], np.rot90(faces[3], 2), np.rot90(faces[2], 2)],
            ]
        )
        faces[curr_face][i, j] = 0
    else:
        faces[curr_face][i, j] = 4
        tmp2 = np.block(
            [
                [faces[0], faces[1], faces[2]],
                [faces[0], faces[6], faces[0]],
                [np.rot90(faces[4], 2), np.rot90(faces[3], 2), faces[0]],
                [np.rot90(faces[5], -1), faces[0], faces[0]],
            ]
        )
        faces[curr_face][i, j] = 0

    i, j = np.argwhere(tmp2 == 4)[0]
    """
    example data:
        i = 4
        j = 6
        curr_face = 4
        move_dir = (0, 1) (så upp roterat...)

        > (i + 1) * 1000 + (j + 1) * 4 + 3

    input data:
        i = 73
        j = 71
        curr_face = 6
        move_dir = (0, 1)  # samma i boards koordinatsystem

        > (i + 1) * 1000 + (j + 1) * 4 + face_score[move_dir]
        74288
    """
    print(sorted(debug_set))
    breakpoint()
    # return int((i + 1) * 1000 + 4 * (j + 1) + face_score[move_dir])


if 0:
    example_answer = b(puzzle.example_data, example=True)
    print(example_answer)
    assert example_answer == 5031
answer = b(puzzle.input_data)
print("b:", answer)
# puzzle.answer_b = answer
