from aocd.models import Puzzle

YEAR = 2025
DAY = 5

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    ranges, ids = data.split("\n\n")
    rr = []
    for r in ranges.split():
        lo, hi = r.split("-")
        rr.append(range(int(lo), int(hi) + 1))
    c = 0
    for id in ids.split():
        for r in rr:
            if int(id) in r:
                c += 1
                break
    return c


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 698

# Part b
def b(data):
    ranges, ids = data.split("\n\n")
    rr = []
    for r in ranges.split():
        lo, hi = r.split("-")
        rr.append(range(int(lo), int(hi) + 1))
    gg = set()
    for id in ids.split():
        for r in rr:
            if int(id) in r:
                gg.add(r)
    gg = sorted(list(gg), key=lambda r: (r.start, r.stop))
    gg = [[g.start, g.stop - 1] for g in gg]
    oo = [gg[0].copy()]
    for g in gg[1:]:
        if g[1] > oo[-1][1]:
            if g[0] <= oo[-1][1]:
                oo[-1][1] = g[1]
            else:
                oo.append(g.copy())
    s = 0
    for g in oo:
        s += g[1] - g[0] + 1
    return s


example_answer = b(example.input_data)
print(example_answer)
assert example_answer == 14
answer = b(puzzle.input_data)
print("b:", answer)
assert answer != 337646673718690
assert answer != 337878930639144
assert answer != 337878930639150
assert answer != 348324210241501
#puzzle.answer_b = answer
