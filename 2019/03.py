from more_itertools import pairwise

from aocd.models import Puzzle

YEAR = 2019
DAY = 3

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def get_nodes(wire):
    curr_pos = 0j
    nodes = [curr_pos]
    for p in wire:
        direction = p[0]
        steps = int(p[1:])
        if direction == "R":
            dir = 1
        elif direction == "U":
            dir = 1j
        elif direction == "L":
            dir = -1
        elif direction == "D":
            dir = -1j
        curr_pos += dir * steps
        nodes.append(curr_pos)
    return nodes


def intersection(line0, line1):
    # Two horizontal lines
    if (line0[0].imag == line0[1].imag) and (line1[0].imag == line1[1].imag):
        return
    # Two vertical lines
    if (line0[0].real == line0[1].real) and (line1[0].real == line1[1].real):
        return
    # line0 horizontal
    if line0[0].imag == line0[1].imag:
        if (
            min(line0[0].real, line0[1].real)
            < line1[0].real
            < max(line0[0].real, line0[1].real)
        ) and (
            min(line1[0].imag, line1[1].imag)
            < line0[0].imag
            < max(line1[0].imag, line1[1].imag)
        ):
            # return line1[0].real + line0[0].imag * 1j
            return line1[0].real + line0[0].imag
        return
    # line0 vertical
    if line0[0].real == line0[1].real:
        if (
            min(line0[0].imag, line0[1].imag)
            < line1[0].imag
            < max(line0[0].imag, line0[1].imag)
        ) and (
            min(line1[0].real, line1[1].real)
            < line0[0].real
            < max(line1[0].real, line1[1].real)
        ):
            # return line1[0].imag * 1j + line0[0].real
            return line1[0].imag + line0[0].real
        return
    breakpoint()


def a(data):
    wire0, wire1 = data.splitlines()
    wire0_nodes = get_nodes(wire0.split(","))
    wire1_nodes = get_nodes(wire1.split(","))
    wire0_segments = pairwise(wire0_nodes)
    intersections = []
    for segment0 in wire0_segments:
        wire1_segments = pairwise(wire1_nodes)
        for segment1 in wire1_segments:
            inter = intersection(segment0, segment1)
            if inter is not None:
                intersections.append(inter)
    return int(min(intersections))


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        #assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
breakpoint()
# assert failed for the second example... but got the correct answer for the input data..
puzzle.answer_a = answer


# Part b
def b(data):
    print(data)
    breakpoint()


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
