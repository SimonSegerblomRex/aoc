import re
from math import prod

import matplotlib.pyplot as plt
import numpy as np
from aocd.models import Puzzle

YEAR = 2024
DAY = 14

puzzle = Puzzle(year=YEAR, day=DAY)


def display_image(positions, width, height, title):
    image = np.zeros((height, width), dtype=int)
    for pos in set(positions):
        image[int(pos.imag), int(pos.real)] = 1
    plt.imshow(image)
    plt.title(title)
    plt.show()


# Part a
def a(data, width=101, height=103, seconds=100, display=False):
    robot_pos = []
    robot_dir = []
    for line in data.splitlines():
        x = [int(n) for n in re.findall(r"(-?\d+)", line)]
        pos = x[0] + x[1] * 1j
        dir = x[2] + x[3] * 1j
        robot_pos.append(pos)
        robot_dir.append(dir)
    for second in range(seconds):
        for i, (pos, dir) in enumerate(zip(robot_pos, robot_dir)):
            new_pos = pos + dir
            new_pos = (new_pos.real % width) + (new_pos.imag % height) * 1j
            robot_pos[i] = new_pos
        if display and np.var(robot_pos) < 1000:
            display_image(robot_pos, width, height, second + 1)
    scores = [0, 0, 0, 0]
    for pos in robot_pos:
        if pos.real < (width - 1) // 2 and pos.imag < (height - 1) // 2:
            scores[0] += 1
        elif pos.real > (width - 1) // 2 and pos.imag < (height - 1) // 2:
            scores[1] += 1
        elif pos.real < (width - 1) // 2 and pos.imag > (height - 1) // 2:
            scores[2] += 1
        elif pos.real > (width - 1) // 2 and pos.imag > (height - 1) // 2:
            scores[3] += 1
    return prod(scores)


answer = a(puzzle.input_data)
print("a:", answer)


# Part b
answer = a(puzzle.input_data, seconds=10000, display=True)
