import re

import numpy as np
from aocd.models import Puzzle

YEAR = 2023
DAY = 21

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, max_steps, start=(65+65j)):
    rocks = []
    for i, line in enumerate(data.splitlines()):
        for m in re.finditer("#", line):
            j = m.start()
            rocks.append(complex(i, j))

    lines = data.splitlines()
    height, width = len(lines), len(lines[0])


    curr_pos = set([(start, 0)])
    #visited = set()
    visited = {}
    dirs = [0 + 1j, -1 + 0j, 0 - 1j, 1 + 0j]
    while curr_pos:
        pos, steps = curr_pos.pop()
        if steps > max_steps:
            continue
        for dir in dirs:
            next_pos = pos + dir
            next_pos_mod = complex(next_pos.real % height, next_pos.imag % width)
            if next_pos_mod in rocks:
                continue
            if next_pos in visited:
                if steps >= visited[next_pos]:
                    continue
            curr_pos.add((next_pos, steps + 1))
        if pos not in visited:
            visited[pos] = steps
        else:
            visited[pos] = min(steps, visited[pos])
        #print(len(visited), len(curr_pos), steps)
    if not max_steps % 2:
        visited = set([pos for pos, s in visited.items() if not s % 2])
    else:
        visited = set([pos for pos, s in visited.items() if s % 2])

    def manhattan(n0, n1):
        return int(abs(n1.real - n0.real) + abs(n1.imag - n0.imag))

    center_rocks = [r for r in rocks if manhattan(r, start) < 65]

    center_rocks = [r for r in rocks if manhattan(r, start) <= 65]

    if not max_steps % 2:
        center_rocks_relevant = [r for r in center_rocks if not (r.real + r.imag) % 2]
    else:
        center_rocks_relevant = [r for r in center_rocks if (r.real + r.imag) % 2]

    if 1:
        # Debug print

        s = 0
        for i in range(height):
            for j in range(width):
                if complex(i, j) in center_rocks_relevant:
                    print("R", end="")
                elif complex(i, j) in center_rocks:
                    print("C", end="")
                elif complex(i, j) in rocks:
                    print("#", end="")
                elif complex(i, j) in visited:
                    print("O", end="")
                    s += 1
                else:
                    print(".", end="")
            print("")
        print(s)
        breakpoint()
    if 1:
        grid2 = np.vstack(
            [np.frombuffer(row.encode(), dtype=np.int8) for row in data.splitlines()]
        )
        grid2[grid2 == ord(".")] = 0
        grid2[grid2 == ord("#")] = 2
        grid2[grid2 == ord("S")] = 0
        tmp = grid2.copy()
        tmp.ravel()[::2] = 1
        tmp[grid2 == 2] = 2

        stones = grid2.copy()
        stones[stones == 1] = 0
        stones[stones == 2] = 1

        s_ul = np.triu(np.rot90(stones[:height//2,:width//2], -1), 1).sum()
        s_ur = np.triu(stones[:height//2,width//2:], 1).sum()
        s_ll = np.tril(stones[height//2:,:width//2], -1).sum()
        s_lr = np.triu(np.rot90(stones[height//2:, width//2:], 1), 1).sum()
        print(s_ul, s_ur, s_ll, s_lr, s_ul+s_ur+s_ll+s_lr)
        # len(rocks) = 2273, stones 1120
        breakpoint()

        tmp2 = grid2.copy()
        tmp2.ravel()[1::2] = 1
        tmp2[grid2 == 2] = 2
        print(tmp2[tmp2 == 1].sum())


        print(tmp[tmp == 1].sum())
        print(tmp2[tmp2 == 1].sum())
        s = 0
        for i in range(height):
            for j in range(width):
                if tmp[i, j] == 2:
                    print("#", end="")
                elif tmp[i, j] == 1:
                    print("O", end="")
                    s += 1
                else:
                    print(".", end="")
            print("")
        breakpoint()
    return len(visited)


# for example in puzzle.examples:
#     if example.answer_a:
#         example_answer = a(example.input_data, max_steps=6)
#         print(f"Example answer: {example_answer} (expecting: {16})")
#         assert example_answer == 16
#answer = a(puzzle.input_data, max_steps=64)
#print("a:", answer)
#assert answer > 3459
#puzzle.answer_a = answer



#a(example.input_data, max_steps=1000)
#a(example.input_data, max_steps=25)

#answer = a(puzzle.input_data, max_steps=196)#, start=130+130j)
#answer = a(puzzle.input_data, max_steps=10)#131+65)
#answer = a(puzzle.input_data, max_steps=65)

# Part b
def b(data):
    #max_steps=65: visited= 3742len(center_rocks_relevant): 613
    steps = 26501365
    return (steps//131+1)**2*3742+(steps//131)**2*3682+(steps//131*2+1)**2//2//2*3709+(steps//131*2+1)**2//2//2*3748


answer = b(puzzle.input_data)
print("b:", answer)
assert answer < 26501365**2
puzzle.answer_b = answer
