from aocd.models import Puzzle

YEAR = 2021
DAY = 10

puzzle = Puzzle(year=YEAR, day=DAY)
example_data = """[({(<(())[]>[[{[]{<()<>>
[(()[<>])]({[<{<<[]>>(
{([(<{}[<>[]}>{[]{[(<()>
(((({<>}<{<{<>}{[]{[]{}
[[<[([]))<([[{}[[()]]]
[{[{({}]{}}([{[{{{}}([]
{<[[]]>}<{[{[{[]{()[[[]
[<(<(<(<{}))><([]([]()
<{([([[(<>()){}]>(<<{{
<{([{{}}[<[[[<>{}]]]>[]]
"""

# Part a
def a(data):
    open_c = "([{<"
    close_c = ")]}>"
    close_to_open = dict(zip(close_c, open_c))
    close_to_points = dict(zip(close_c, (3, 57, 1197, 25137)))
    corrupt = []
    for line in data.splitlines():
        o = []
        for c in line:
            if c in open_c:
                o.append(c)
            elif c in close_c:
                if close_to_open[c] != o.pop():
                    corrupt.append(c)
                    break
    return sum([close_to_points[c] for c in corrupt])


example_answer = a(example_data)
print(example_answer)
assert example_answer == 26397
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 311949


# Part b
def b(data):
    open_c = "([{<"
    close_c = ")]}>"
    close_to_open = dict(zip(close_c, open_c))
    open_to_points = dict(zip(open_c, (1, 2, 3, 4)))
    missing = []
    scores = []
    for line in data.splitlines():
        o = []
        for c in line:
            if c in open_c:
                o.append(c)
            elif c in close_c:
                if close_to_open[c] != o.pop():
                    break
        else:
            total = 0
            for c in o[::-1]:
                total *= 5
                total += open_to_points[c]
            scores.append(total)
    return sorted(scores)[(len(scores) - 1) // 2]


example_answer = b(example_data)
print(example_answer)
assert example_answer == 288957
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 3042730309
