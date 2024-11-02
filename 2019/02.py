import numpy as np
from aocd.models import Puzzle

YEAR = 2019
DAY = 2

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    codes = list(map(int, data.split(",")))
    codes[1] = 12
    codes[2] = 2
    pos = 0
    while True:
        opcode = codes[pos]
        if opcode == 1:
            codes[codes[pos + 3]] = codes[codes[pos + 1]] + codes[codes[pos + 2]]
            pos += 4
        elif opcode == 2:
            codes[codes[pos + 3]] = codes[codes[pos + 1]] * codes[codes[pos + 2]]
            pos += 4
        elif opcode == 99:
            break
        else:
            print(f"Unknown {opcode=}")
            breakpoint()
    return codes[0]


answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 3716293


# Part b
def b(data):
    codes_orig = list(map(int, data.split(",")))
    for noun in range(100):
        for verb in range(100):
            codes = codes_orig.copy()
            codes[1] = noun
            codes[2] = verb
            pos = 0
            if noun == 99:
                print(noun, verb)
            try:
                while True:
                    opcode = codes[pos]
                    if opcode == 1:
                        codes[codes[pos + 3]] = codes[codes[pos + 1]] + codes[codes[pos + 2]]
                        pos += 4
                    elif opcode == 2:
                        codes[codes[pos + 3]] = codes[codes[pos + 1]] * codes[codes[pos + 2]]
                        pos += 4
                    elif opcode == 99:
                        break
                    else:
                        break
            except IndexError:
                continue
            if codes[0] == 19690720:
                return 100 * noun + verb


answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 6429
