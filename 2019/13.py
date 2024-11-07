from collections import defaultdict
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
from aocd.models import Puzzle

YEAR = 2019
DAY = 13

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def run(codes, inputs, pos=0, relative_base=0):
    while True:
        instruction = codes[pos]
        opcode = abs(instruction) % 100
        finished = False
        def get_mode(instruction, i):
            return int(("0000" + str(instruction))[-3 - i])
        def get_idx(instruction, i):
            mode = get_mode(instruction, i)
            if mode == 0:
                return codes[pos + i + 1]
            if mode == 1:
                return pos + i + 1
            if mode == 2:
                return relative_base + codes[pos + i + 1]
            raise NotImplementedError(f"Unsupported {mode=}")
        if opcode == 1:
            idx0 = get_idx(instruction, 0)
            idx1 = get_idx(instruction, 1)
            idx2 = get_idx(instruction, 2)
            codes[idx2] = codes[idx0] + codes[idx1]
            pos += 4
        elif opcode == 2:
            idx0 = get_idx(instruction, 0)
            idx1 = get_idx(instruction, 1)
            idx2 = get_idx(instruction, 2)
            codes[idx2] = codes[idx0] * codes[idx1]
            pos += 4
        elif opcode == 3:
            idx0 = get_idx(instruction, 0)
            codes[idx0] = inputs.pop(0)
            pos += 2
        elif opcode == 4:
            idx0 = get_idx(instruction, 0)
            inputs.append(codes[idx0])
            pos += 2
            break
        elif opcode == 5:
            idx0 = get_idx(instruction, 0)
            idx1 = get_idx(instruction, 1)
            if codes[idx0] != 0:
                pos = codes[idx1]
            else:
                pos += 3
        elif opcode == 6:
            idx0 = get_idx(instruction, 0)
            idx1 = get_idx(instruction, 1)
            if codes[idx0] == 0:
                pos = codes[idx1]
            else:
                pos += 3
        elif opcode == 7:
            idx0 = get_idx(instruction, 0)
            idx1 = get_idx(instruction, 1)
            idx2 = get_idx(instruction, 2)
            if codes[idx0] < codes[idx1]:
                codes[idx2] = 1
            else:
                codes[idx2] = 0
            pos += 4
        elif opcode == 8:
            idx0 = get_idx(instruction, 0)
            idx1 = get_idx(instruction, 1)
            idx2 = get_idx(instruction, 2)
            if codes[idx0] == codes[idx1]:
                codes[idx2] = 1
            else:
                codes[idx2] = 0
            pos += 4
        elif opcode == 9:
            idx0 = get_idx(instruction, 0)
            relative_base += codes[idx0]
            pos += 2
        elif opcode == 99:
            pos += 1
            finished = True
            break
        else:
            print(f"Unknown {opcode=}")
            breakpoint()
    return inputs.pop(), pos, finished, relative_base


def a(data):
    codes = list(map(int, data.split(",")))
    codes.extend([0]*1000)
    tile_info = []
    cpos = 0
    relative_base = 0
    finished = False
    while not finished:
        info, cpos, finished, relative_base = run(codes, [0], cpos, relative_base)
        tile_info.append(info)
    tile_info = tile_info[:-1]
    tile_info = zip(tile_info[::3], tile_info[1::3], tile_info[2::3])
    tile_info = (tile for tile in tile_info if tile[2] == 2)
    return len(list(tile_info))


answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 324


plt.ion()

# Part b
def b(data):
    codes = list(map(int, data.split(",")))
    codes.extend([0]*1000)
    codes[0] = 2
    cpos = 0
    relative_base = 0
    finished = False
    inp = 0
    tmp = []
    score = 0
    wall = []
    blocks = []
    paddle = 24 + 24j
    ball = 0 + 0j
    all_set = False
    first_run = True
    dir = 5
    goal = 24
    plot = False
    while not finished:
        diff = goal - int(paddle.real)
        if diff > 0:
            inp = 1
        elif diff < 0:
            inp = -1
        else:
            inp = 0
        out, cpos, finished, relative_base = run(codes, [inp], cpos, relative_base)
        tmp.append(out)
        if len(tmp) == 3:
            if tmp[0] == -1 and tmp[1] == 0:
                score = tmp[2]
            elif tmp[2] == 0:
                pass
            elif tmp[2] == 1:
                wall.append(tmp[0] + tmp[1] * 1j)
            elif tmp[2] == 2:
                blocks.append(tmp[0] + tmp[1] * 1j)
            elif tmp[2] == 3:
                paddle = tmp[0] + tmp[1] * 1j
                wall_x_max = int(max(n.real for n in wall))
                wall_y_max = int(max(n.imag for n in wall))
            elif tmp[2] == 4:
                ball_new_pos = tmp[0] + tmp[1] * 1j
                dir = ball_new_pos -ball
                ball = ball_new_pos
                if abs(dir) < 2:
                    if plot:
                        if first_run:
                            image = np.zeros((wall_y_max + 2, wall_x_max + 1))
                            image[[int(n.imag) for n in wall], [int(n.real) for n in wall]] = 1
                        image[[int(n.imag) for n in blocks], [int(n.real) for n in blocks]] = 2
                        image[int(paddle.imag), int(paddle.real)] = 3
                        image[int(ball.imag), int(ball.real)] = 4
                        if first_run:
                            pp = plt.imshow(image)
                        plt.title(f"score: {score}, goal pos: {goal}")
                        pp.set_data(image)
                        plt.draw()
                        plt.pause(0.5)
            tmp = []
            goal = int(ball.real)
    return score


answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 15957
