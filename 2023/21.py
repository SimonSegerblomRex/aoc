import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, max_steps):
    rocks = []
    start = None
    for i, line in enumerate(data.splitlines()):
        for m in re.finditer("#", line):
            j = m.start()
            rocks.append(complex(i, j))
        if start is None:
            if (m := re.search("S", line)) is not None:
                start = complex(i, m.start())

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
    visited = set([pos for pos, s in visited.items() if not s % 2])
    breakpoint()
    if 1:
        # Debug print

        s = 0
        for i in range(height):
            for j in range(width):
                if complex(i, j) in rocks:
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

#answer = a(puzzle.input_data, max_steps=65)
answer = a(puzzle.input_data, max_steps=10)#131+65)
#answer = a(puzzle.input_data, max_steps=65)

# Part b
def b(data):
    #26501365 % 131 = 65
    #len(visited) a med input 131 innanför rutan: 16054
    #len(rocks) 2273
    #
    #
    # len(visited) a med input 131: 7474
    #
    # 7474-3682=3792
    # 7479 - 3682 = 3797
    # 7474
    #
    #len(rocks) 2273
    #26501365//131 = 202300
    # .O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.
    #print(".O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O".count("O")) 65
    #202300 *
    #   (26501365 * 2 + 1)**2
    #return (26501365//131*2+1)*3682+(26501365//131*2-1)*12372
    #>>> (26501365-65)/131
    #202300.0
    #(202300*2+1)**2//2
    #((26501365*2+1)//131)**2//2=81850984600
    #print(data)
    #return 81850984601*3682 + 81850984600*3792 + 1
    #return 81850984601*7479
    #return 81850984601*7479+1
    #return 81850984601*3682 + 81850984600*3797 + 1
    #return ((26501365//131*2+1)**2//2+1)*7479+1
    #return ((26501365//131*2+1)**2//2+1)*7474+1
    #
    #a(puzzle.input_data, max_steps=131+65)
    #(Pdb) len(visited)
    #33564 (exklusive S)
    #
    #(Pdb) tmp[tmp == 1].sum()
    #7479
    #(Pdb) tmp2[tmp2 == 1].sum()
    #7409
    breakpoint()


answer = b(puzzle.input_data)
print("b:", answer)
assert answer > 6495439710
#assert answer < 1314035706768400
#assert answer < 913129584201282
#                611754258907874
assert answer < 26501365**2
assert answer != 612163513830880
puzzle.answer_b = answer
