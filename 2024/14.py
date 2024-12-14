import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, width=101, height=103, seconds=100, display=False):
    robot_pos = []
    robot_dir = []
    for line in data.splitlines():
        x = [int(n) for n in re.findall(r"(-?\d+)", line)]
        pos = x[0] + x[1]*1j
        dir = x[2] + x[3]*1j
        robot_pos.append(pos)
        robot_dir.append(dir)
    for second in range(seconds):
        for i, (pos, dir) in enumerate(zip(robot_pos, robot_dir)):
            new_pos = pos + dir
            new_pos = (new_pos.real % width) + (new_pos.imag % height) * 1j
            robot_pos[i] = new_pos
        if display and np.var(robot_pos) < 1000:
            import matplotlib.pyplot as plt
            image = np.zeros((height, width), dtype="u1")
            tmp = set(robot_pos)
            for pos in tmp:
                image[int(pos.imag), int(pos.real)] = 1
            plt.imshow(image)
            plt.title(second + 1)
            plt.show()
    scores = [0, 0, 0 ,0]
    for pos in robot_pos:
        if (pos.real < ((width - 1) // 2)) and (pos.imag < ((height - 1) // 2)):
            scores[0] += 1
        elif (pos.real > ((width - 1) // 2)) and (pos.imag < ((height - 1) // 2)):
            scores[1] += 1
        elif (pos.real < ((width - 1) // 2)) and pos.imag > (((height - 1) // 2)):
            scores[2] += 1
        elif (pos.real > ((width - 1) // 2)) and pos.imag > (((height - 1) // 2)):
            scores[3] += 1
    print(scores)
    return scores[0] * scores[1] * scores[2] * scores[3]


example = """p=0,4 v=3,-3
p=6,3 v=-1,-3
p=10,3 v=-1,2
p=2,0 v=2,-1
p=0,0 v=1,3
p=3,0 v=-2,-2
p=7,6 v=-1,-3
p=3,0 v=-1,-2
p=9,3 v=2,3
p=7,3 v=-1,2
p=2,4 v=2,-3
p=9,5 v=-3,-3"""
answer = a(example, width=11, height=7)
print(answer)
assert answer == 12
answer = a(puzzle.input_data)
print("a:", answer)


# Part b
answer = a(puzzle.input_data, seconds=1000000, display=True)
print("b:", answer)
#puzzle.answer_b = answer
