import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    grid, moves = data.split("\n\n")
    wall = set()
    boxes = set()
    pos = None
    for i, line in enumerate(grid.splitlines()):
        width = len(line)
        for j, c in enumerate(line):
            if c == "#":
                wall.add(j + i*1j)
            elif c == "O":
                boxes.add(j + i*1j)
            elif c == "@":
                pos = j + i*1j
    height = i + 1
    move2dir = {
        ">": 1,
        "^": -1j,
        "<": -1,
        "v": 1j,
    }
    for move in moves.replace("\n", ""):
        dir = move2dir[move]
        if pos + dir in wall:
            continue
        if pos + dir not in boxes:
            pos += dir
            continue
        boxes_to_move = 0
        pos_to_check = pos + dir
        while True:
            if pos_to_check in boxes:
                boxes_to_move += 1
                pos_to_check += dir
            elif pos_to_check in wall:
                boxes_to_move = 0
                break
            else:
                break
        if boxes_to_move > 0:
            pos += dir
            box_to_move = pos
            boxes.remove(pos)
            boxes.add(pos + boxes_to_move * dir)
    s = 0
    for box in boxes:
        s += box.real + 100 * box.imag
    return s
    breakpoint()


answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    grid, moves = data.split("\n\n")
    grid = grid.replace("#", "##")
    grid = grid.replace("O", "[]")
    grid = grid.replace(".", "..")
    grid = grid.replace("@", "@.")
    wall = set()
    boxes = set()
    box_coords = set()
    pos = None
    for i, line in enumerate(grid.splitlines()):
        width = len(line)
        for j, c in enumerate(line):
            if c == "#":
                wall.add(j + i*1j)
                wall.add(j + 0.5 + i*1j)
            elif c == "[":
                boxes.add(j + i*1j)
                box_coords.add(j + 0.5 + i*1j)
            elif c == "]":
                boxes.add(j + i*1j)
            elif c == "@":
                pos = j + i*1j
    height = i + 1
    move2dir = {
        ">": 1,
        "^": -1j,
        "<": -1,
        "v": 1j,
    }
    for move in moves.replace("\n", ""):
        boxes = set()
        for box_coord in box_coords:
            boxes.add(box_coord - 0.5)
            boxes.add(box_coord + 0.5)
        if 0:
            for i in range(height):
                for j in range(width):
                    p = j + i*1j
                    if p in wall:
                        if int(p.real) == p.real:
                            print("#", end="")
                    elif p == pos:
                        print("@", end="")
                    elif p - 0.5 in box_coords:
                        print("[]", end="")
                    elif p + 0.5 in box_coords:
                        pass
                    else:
                        print(".", end="")
                print()
            print(move)
            breakpoint()
        dir = move2dir[move]
        if pos + dir in wall:
            continue
        if pos + dir not in boxes:
            pos += dir
            continue
        boxes_to_move = 0
        pos_to_check = pos + dir
        while True:
            if pos_to_check in boxes:
                boxes_to_move += 1
                pos_to_check += dir
            elif pos_to_check in wall:
                boxes_to_move = 0
                break
            else:
                break
        if boxes_to_move > 0:
            if dir in (1, -1):
                box_coord_to_move = pos + dir * 1.5
                pos += dir
                for i in range(boxes_to_move // 2):
                    if box_coord_to_move not in box_coords:
                        break
                    box_coords.remove(box_coord_to_move)
                    box_coords.add(box_coord_to_move + dir)
                    box_coord_to_move += 2*dir
                continue
            boxes_to_move = set()
            pot_boxes = set([pos + dir - 0.5, pos + dir + 0.5])
            box = (pot_boxes & box_coords).pop()
            boxes_to_move.add(box)
            can_move = True
            while True:
                tmp = boxes_to_move.copy()
                for box in tmp:
                    cand0 = box + dir - 1
                    cand1 = box + dir + 1
                    cand2 = box + dir
                    if cand0 in box_coords:
                        boxes_to_move.add(cand0)
                    if cand1 in box_coords:
                        boxes_to_move.add(cand1)
                    if cand2 in box_coords:
                        boxes_to_move.add(cand2)
                    for cand in (cand2,):
                        if (cand - 0.5) in wall or (cand + 0.5) in wall:
                            can_move = False
                    if not can_move:
                        break
                if tmp == boxes_to_move:
                    break
            if can_move:
                pos += dir
                new_box_coords = set()
                for box in boxes_to_move:
                    new_box_coords.add(box + dir)
                    box_coords.remove(box)
                box_coords |= new_box_coords
    return sum(box.real - 0.5 + 100 * box.imag for box in box_coords)


example0 = """#######
#...#.#
#.....#
#..OO@#
#..O..#
#.....#
#######

<vv<<^^<<^^"""

example = """##########
#..O..O.O#
#......O.#
#.OO..O.O#
#..O@..O.#
#O#..O...#
#O..O..O.#
#.OO.O.OO#
#....O...#
##########

<vv>^<v^>v>^vv^v>v<>v^v<v<^vv<<<^><<><>>v<vvv<>^v^>^<<<><<v<<<v^vv^v>^
vvv<<^>^v^^><<>>><>^<<><^vv^^<>vvv<>><^^v>^>vv<>v<<<<v<^v>^<^^>>>^<v<v
><>vv>v^v^<>><>>>><^^>vv>v<^^^>>v^v^<^^>v^^>v^<^v>v<>>v^v^<v>v^^<^^vv<
<<v<^>>^^^^>>>v^<>vvv^><v<<<>^^^vv^<vvv>^>v<^^^^v<>^>vvvv><>>v^<<^^^^^
^><^><>>><>^^<<^^v>>><^<v>^<vv>>v>>>^v><>^v><<<<v>>v<v<v>vvv>^<><<>^><
^>><>^v<><^vvv<^^<><v<<<<<><^v<<<><<<^^<v<^^^><^>>^<v^><<<^>>^v<v^v<v^
>^>>^v>vv>^<<^v<>><<><<v<<v><>v<^vv<<<>^^v^>^^>>><<^v>>v^v><^^>>^<>vv^
<><^^>^^^<><vvvvv^v<v<<>^v<v>v<<^><<><<><<<^^<<<^<<>><<><^^^>^^<>^>v<>
^^>vv<^v^v<vv>^<><v<^v>^^^>>>^^vvv^>vvv<>>>^<^>>>>>^<<^v>^vvv<>^<><<v>
v^^>>><<^^<>>^v^<v^vv<>v^<<>^<^v^v><^<<<><<^<v><v<>vv>>v><v^<vv<>v^<<^"""
if 1:
    answer = b(example0)
    print("b:", answer)
answer = b(example)
print("b:", answer)
assert answer == 9021
answer = b(puzzle.input_data)
print("b:", answer)
assert answer > 1518399
puzzle.answer_b = answer
