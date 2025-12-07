from aocd.models import Puzzle

YEAR = 2025
DAY = 7

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    splitters = set()
    for j, line in enumerate(data.splitlines()):
        for i, c in enumerate(line):
            if c == "S":
                S = i + 1j * j
            elif c == "^":
                splitters.add(i + 1j * j)
    max_j = j
    beam_fronts = set([S + 1j])
    curr_j = S.imag
    c = 0
    while curr_j < max_j:
        new_fronts = set()
        for front in beam_fronts:
            if front + 1j in splitters:
                new_fronts.add(front - 1 + 1j)
                new_fronts.add(front + 1 + 1j)
                c += 1
            else:
                new_fronts.add(front + 1j)
        curr_j = front.imag
        beam_fronts = new_fronts
    return c


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 1649


# Part b
def b(data):
    splitters = set()
    for j, line in enumerate(data.splitlines()):
        for i, c in enumerate(line):
            if c == "S":
                S = i + 1j * j
            elif c == "^":
                splitters.add(i + 1j * j)
    stop = j
    cache = {}

    def paths(front):
        if front in cache:
            return cache[front]
        if front.imag > stop:
            return 1
        c = 0
        if front + 1j in splitters:
            c += paths(front - 1 + 1j)
            c += paths(front + 1 + 1j)
        else:
            c += paths(front + 1j)
        cache[front] = c
        return c

    return paths(S)


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 16937871060075
