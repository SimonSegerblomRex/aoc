from aocd.models import Puzzle

YEAR = 2022
DAY = 6

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data, pl):
    marker = ""
    for i, c in enumerate(data):
        marker += c
        if len(marker) == pl:
            l = pl
            while len(set(marker)) < l:
                marker = marker[1:]
                l -= 1
            if len(marker) == pl:
                break
    return i + 1


example_answer = a(puzzle.example_data, pl=4)
print(example_answer)
assert example_answer == 7
answer = a(puzzle.input_data, pl=4)
print("a:", answer)
assert answer == 1757

# Part b
example_answer = a(puzzle.example_data, pl=14)
print(example_answer)
answer = a(puzzle.input_data, pl=14)
print("b:", answer)
assert answer == 2950
