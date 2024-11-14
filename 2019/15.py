import queue
from collections import defaultdict
from itertools import permutations
from operator import itemgetter

import numpy as np
from aocd.models import Puzzle

YEAR = 2019
DAY = 15

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


def plot(*args):
    import numpy as np
    arrays = []
    for arg in args:
        arrays.append(np.array(arg).astype(complex))
    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0
    for array in arrays:
        min_x = min(array.real.min(), min_x)
        min_y = min(array.imag.min(), min_y)
        max_x = max(array.real.max(), max_x)
        max_y = max(array.imag.max(), max_y)
    for array in arrays:
        array -= min_x + min_y*1j
    max_x -= min_x
    max_y -= min_y
    min_x = 0
    min_y = 0
    import matplotlib.pyplot as plt
    image = np.zeros((int(max_y) + 1, int(max_x) + 1), dtype=int)
    for c, array in enumerate(arrays, 1):
        image[array.imag.astype(int), array.real.astype(int)] = c
    plt.imshow(image)
    plt.show()


def ab(data, a=False):
    codes = list(map(int, data.split(",")))
    codes.extend([0]*10)
    codes_orig = codes.copy()
    cpos = 0
    relative_base = 0
    finished = False
    move = 1
    dir2move_map = {1j: 1, -1j: 2, -1: 3, 1: 4}
    start = 0 + 0j
    pos = start
    dir = 1j
    path = []
    wall = set()
    dead_end = []
    oxygen = []
    def neighbours(p):
        return [p + 1, p + 1j, p - 1, p -1j]
    i = 0
    while not finished:
        i = (i + 1) % 100
        move = dir2move_map[dir]
        status, cpos, finished, relative_base = run(codes, [move], cpos, relative_base)
        if status == 0:
            # hit wall
            wall.add(pos + dir)
        elif status == 1:
            # moved one step
            pos += dir
            path.append(pos)
        elif status == 2:
            # found station
            pos += dir
            oxygen.append(pos)
            break
        neigh = neighbours(pos)
        candidates = [n for n in neigh if n not in wall and n not in dead_end and n not in path]
        if not candidates:
            # no candidate...dead end... find way out
            candidates = [n for n in neigh if n not in wall and n not in dead_end]
            new_pos = candidates[0]
            dead_end.append(pos)
        else:
            # go to neighbour closest to start not in path
            dist = [(abs(p - start), p) for p in candidates]
            new_pos = dist[0][1]
        dir = new_pos - pos
        if 0:#i == 0:
            plot(list(wall), path, [pos])

    # find shortest path back to start...(A*, copied from 2023/17)
    goal = start
    start = pos

    def h(node):
        return int(abs(goal.real - node.real) + abs(goal.imag - node.imag))

    g_score = defaultdict(lambda: 1e12)
    g_score[start] = 0

    f_score = defaultdict(lambda: 1e12)
    f_score[start] = h(start)

    open_set = queue.PriorityQueue()
    open_set.put((f_score[start], 0, start))

    intcode_state = {}
    intcode_state[start] = (codes.copy(), cpos, relative_base)

    i = 1
    while not open_set.empty():
        _, _, current = open_set.get()
        if current == goal:
            answer_a = int(g_score[current])
            break

        dirs = [0 + 1j, -1 + 0j, 0 - 1j, 1 + 0j]
        for dir in dirs:
            pos = current
            codestmp, cpos, relative_base = intcode_state[current]
            codes = codestmp.copy()
            move = dir2move_map[dir]
            status, cpos, finished, relative_base = run(codes, [move], cpos, relative_base)
            if status == 0:
                # hit wall
                continue
            pos += dir
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score[pos]:
                g_score[pos] = tentative_g_score
                f_score[pos] = tentative_g_score + h(pos)
                open_set.put((f_score[pos], i, pos))
                intcode_state[pos] = (codes.copy(), cpos, relative_base)
                i += 1

    if a:
        return answer_a

    # b...
    # back to oxygen pos
    pos = oxygen[0]
    codes, cpos, relative_base = intcode_state[pos]

    # spawn robots in all directions
    robot_cache = {}

    dirs = [0 + 1j, -1 + 0j, 0 - 1j, 1 + 0j]
    for dir in dirs:
        robot_cache[(pos, dir)] = codes.copy(), cpos, relative_base

    i = 0
    while robot_cache:
        for pos, dir in robot_cache.copy():
            codes, cpos, relative_base = robot_cache[(pos, dir)]
            move = dir2move_map[dir]
            status, cpos, finished, relative_base = run(codes, [move], cpos, relative_base)
            del robot_cache[(pos, dir)]
            if status == 0:
                wall.add(pos + dir)
                continue
            pos += dir
            oxygen.append(pos)
            neigh = neighbours(pos)
            candidates = [n for n in neigh if n not in oxygen and n not in wall]
            for new_pos in candidates:
                robot_cache[(pos, new_pos - pos)] = codes.copy(), cpos, relative_base
        i += 1
    #plot(list(wall), oxygen)
    return i


answer = ab(puzzle.input_data, a=True)
print("a:", answer)
assert answer == 366

answer = ab(puzzle.input_data)
print("b:", answer)
assert answer == 384
