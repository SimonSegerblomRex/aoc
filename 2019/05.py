import numpy as np
from aocd.models import Puzzle

YEAR = 2019
DAY = 5

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    codes = list(map(int, data.split(",")))
    pos = 0
    inp = 1
    while True:
        instruction = codes[pos]
        opcode = instruction % 100
        if opcode == 1:
            idx0 = pos + 1 if (instruction // 100) % 2 else codes[pos + 1]
            idx1 = pos + 2 if (instruction // 1000) % 2 else codes[pos + 2]
            idx2 = pos + 3 if (instruction // 10000) % 2 else codes[pos + 3]
            codes[idx2] = codes[idx0] + codes[idx1]
            pos += 4
        elif opcode == 2:
            idx0 = pos + 1 if (instruction // 100) % 2 else codes[pos + 1]
            idx1 = pos + 2 if (instruction // 1000) % 2 else codes[pos + 2]
            idx2 = pos + 3 if (instruction // 10000) % 2 else codes[pos + 3]
            codes[idx2] = codes[idx0] * codes[idx1]
            pos += 4
        elif opcode == 3:
            idx0 =  pos + 1 if (instruction // 100) % 2 else codes[pos + 1]
            codes[idx0] = inp
            pos += 2
        elif opcode == 4:
            idx0 =  pos + 1 if (instruction // 100) % 2 else codes[pos + 1]
            inp = codes[idx0]
            pos += 2
        elif opcode == 99:
            break
        else:
            print(f"Unknown {opcode=}")
            breakpoint()
    return inp


answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer

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
