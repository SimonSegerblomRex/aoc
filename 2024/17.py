import re

import numpy as np
from aocd.models import Puzzle

YEAR = 2024
DAY = 17

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def run(codes, A, B, C):
    pos = 0
    def combo_operand2val(operand):
        if 0 <= operand <= 3:
            return operand
        if operand == 4:
            return A
        if operand == 5:
            return B
        if operand == 6:
            return C
        print("nej...")
        breakpoint()
    out = []
    while True:
        if pos >= len(codes):
            break
        code = codes[pos]
        if code == 0:
            A //= 2**combo_operand2val(codes[pos + 1])
            pos += 2
        elif code == 1:
            B ^= codes[pos + 1]
            pos += 2
        elif code == 2:
            B = combo_operand2val(codes[pos + 1]) % 8
            pos += 2
        elif code == 3:
            if A != 0:
                pos = codes[pos + 1]
            else:
                pos += 2
        elif code == 4:
            B ^= C
            pos += 2
        elif code == 5:
            out.append(combo_operand2val(codes[pos + 1]) % 8)
            pos += 2
        elif code == 6:
            B = A // 2**combo_operand2val(codes[pos + 1])
            pos += 2
        elif code == 7:
            C = A // 2**combo_operand2val(codes[pos + 1])
            pos += 2
        else:
            print("hmm")
            breakpoint()
    return out


def a(data):
    numbers = [int(n) for n in re.findall(r"(\d+)", data)]
    A = numbers[0]
    B = numbers[1]
    C = numbers[2]
    codes = numbers[3:]
    out = run(codes, A, B, C)
    return ",".join(str(n) for n in out)


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    numbers = [int(n) for n in re.findall(r"(\d+)", data)]
    A = numbers[0]
    B = numbers[1]
    C = numbers[2]
    codes = numbers[3:]
    A = 0
    checking = 1
    incr = 1
    candidates = [A]
    while True:
        new = []
        for A in sorted(candidates):
            for _ in range(1024 if checking == 1 else 8):
                out = run(codes, A, B, C)
                if out == codes:
                    return A
                elif out[:checking] == codes[:checking]:
                    new.append(A)
                A += incr
        candidates = new
        incr <<= 10 if checking == 1 else 3
        checking += 1


answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 105875099912602
