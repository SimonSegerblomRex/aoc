import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

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
    incr = 1
    checking = 1
    candidates = [A]
    for A in candidates:
        new = []
        for A in range(A, A + 1 << 10):
            out = run(codes, A, B, C)
            if out[:checking] == codes[:checking]:
                new.append(A)
    candidates = new
    checking += 1
    good = []
    for A in candidates:
        checking = 2
        incr = 1 << 10
        tmp = [A]
        while True:
            new = []
            for A in tmp:
                for _ in range(8):
                    out = run(codes, A, B, C)
                    if out == codes:
                        good.append(A)
                        break
                    if out[:checking] == codes[:checking]:
                        new.append(A)
                    A += incr
            if out == codes:
                good.append(A)
                break
            if not new:
                break
            tmp = new.copy()
            checking += 1
            incr <<= 3
    return min(good)


if 0:
    for example in puzzle.examples:
        if example.answer_b:
            example_answer = b(example.input_data)
            print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
            assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 105875099912602
