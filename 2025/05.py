from aocd.models import Puzzle

YEAR = 2025
DAY = 5

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    ranges, ids = data.split("\n\n")
    all_ranges = set()
    for r in ranges.split():
        lo, hi = r.split("-")
        all_ranges.add(range(int(lo), int(hi) + 1))
    c = 0
    for id in ids.split():
        for r in all_ranges:
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
    all_ranges = set()
    for r in ranges.split():
        lo, hi = r.split("-")
        all_ranges.add(range(int(lo), int(hi) + 1))
    all_ranges = sorted(list(all_ranges), key=lambda r: (r.start, r.stop))
    all_ranges = [[r.start, r.stop - 1] for r in all_ranges]
    union = [all_ranges[0]]
    for g in all_ranges[1:]:
        if g[1] > union[-1][1]:
            if g[0] <= union[-1][1]:
                union[-1][1] = g[1]
            else:
                union.append(g)
    s = 0
    for g in union:
        s += g[1] - g[0] + 1
    return s


example_answer = b(example.input_data)
print(example_answer)
assert example_answer == 14
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 352807801032167
