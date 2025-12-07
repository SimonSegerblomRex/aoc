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
                S = (i, j)
            elif c == "^":
                splitters.add((i, j))
    max_j = j
    beam_fronts = set([S])
    curr_j = S[1]
    c = 0
    while curr_j < max_j:
        new_fronts = set()
        for i, j in beam_fronts:
            if (i, j + 1) in splitters:
                new_fronts.add((i - 1, j + 1))
                new_fronts.add((i + 1, j + 1))
                c += 1
            else:
                new_fronts.add((i, j + 1))
        curr_j = j
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
                S = (i, j)
            elif c == "^":
                splitters.add((i, j))
    stop = j
    cache = {}

    def paths(coord):
        if coord in cache:
            return cache[coord]
        i, j = coord
        if j > stop:
            return 1
        c = 0
        if (i, j + 1) in splitters:
            c += paths((i - 1, j + 1))
            c += paths((i + 1, j + 1))
        else:
            c += paths((i, j + 1))
        cache[coord] = c
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
