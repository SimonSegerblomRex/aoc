import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def run(codes, A, B, C, special=False):
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
            if special:
                for i, o in enumerate(out):
                    if o != codes[i]:
                        return out
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
    # 105875099939593 too high
    """
    0b11000000100101011110111101110001000111100001001 105875099913993
    0b11000000100101011110111101110001000111100000001 105875099913985
    0b11000000100101011110111101110001000100110011010 105875099912602
    """
    A = 0b10001000100110011010
    check = 2
    incr = -1
    candidates = []
    while True:
        out = run(codes, A, B, C, special=True)
        if out == codes:
            print(bin(A), A)
        if out[:check - 1] == codes[:check - 1]:
            check += 2
            tmp = A
            c = 0
            while tmp:
                c += 1
                tmp >>= 1
            incr = 1 << c
            incr >>= 1
            print(check)
            print(out, codes, check)
        A += incr
    return A


if 0:
    for example in puzzle.examples:
        if example.answer_b:
            example_answer = b(example.input_data)
            print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
            assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
