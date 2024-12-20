import re
from collections import defaultdict

from aocd.models import Puzzle

YEAR = 2019
DAY = 17

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


def a(data):
    codes = list(map(int, data.split(",")))
    codes.extend([0]*1000000)
    cpos = 0
    relative_base = 0
    finished = False
    output = []
    path = set()
    x = 0
    y = 0
    while not finished:
        code, cpos, finished, relative_base = run(codes, [0], cpos, relative_base)
        if chr(code) == "#":
            path.add(x + y*1j)
        x = x + 1
        if chr(code) == "\n":
            x = 0
            y += 1
        output.append(code)
    for c in output:
        print(chr(c), end="")
    def count_neighborus(p):
        neighbours = set((p + 1, p + 1j, p - 1, p - 1j))
        return len(neighbours & path)
    s = 0
    for p in path:
        if count_neighborus(p) == 4:
            s += p.imag * p.real
    return int(s)


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


def b(data):
    codes = list(map(int, data.split(",")))
    codes.extend([0]*1000000)
    cpos = 0
    relative_base = 0
    finished = False
    output = []
    path = set()
    x = 0
    y = 0
    while not finished:
        code, cpos, finished, relative_base = run(codes, [0], cpos, relative_base)
        if chr(code) == "#":
            path.add(x + y*1j)
        if chr(code) in ">^<v":
            start = x + y*1j
            dir_map = {">": 1, "^": -1j, "<": -1, "v": 1j}
            dir = dir_map[chr(code)]
        x = x + 1
        if chr(code) == "\n":
            x = 0
            y += 1
        output.append(code)
    for c in output:
        print(chr(c), end="")
    def get_neighborus(p):
        neighbours = set((p + 1, p + 1j, p - 1, p - 1j))
        return neighbours & path
    pos = start
    def dir_to_rot(new_dir, dir):
        if dir * 1j == new_dir:
            return "R"
        return "L"
    c = 0
    instructions = []
    while True:
        if (pos + dir) not in path:
            neigh = get_neighborus(pos)
            if pos - dir in neigh:
                neigh.remove(pos - dir)
            if not neigh:
                instructions.append(str(c))
                break
            new_dir = list(neigh)[0] - pos
            rot = dir_to_rot(new_dir, dir)
            if c > 0:
                instructions.append(str(c))
            instructions.append(rot)
            c = 0
            dir = new_dir
        c += 1
        pos += dir
    instructions = ["".join(instructions[i:i + 2]) for i in range(0, len(instructions), 2)]
    instructions_all = "".join(instructions)
    # pdb print debug manually finding substrings...
    aa = "L6R12R8"
    bb = "R8R12L12"
    cc = "R12L12L4L4"
    assert aa + bb + bb + aa + cc + aa + cc + aa + cc + bb == instructions_all
    main = "ABBACACACB"
    inp = list(",".join(main))
    inp.append("\n")
    for ll in (aa, bb, cc):
        mm = re.findall("([RL])(\d+)", ll)
        for m in mm:
            inp.append(m[0])
            inp.append(",")
            inp.append(m[1])
            inp.append(",")
        inp[-1] = "\n"
    # run program
    codes = list(map(int, data.split(",")))
    codes.extend([0]*1000000)
    codes[0] = 2
    cpos = 0
    relative_base = 0
    finished = False
    inp.extend(["n", "\n"])
    inp = "".join(inp)
    inp = list(map(int, inp.encode("ascii")))
    while not finished:
        out, cpos, finished, relative_base = run(codes, inp, cpos, relative_base)
        if out > 1000:
            break
    return out


answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 742673
