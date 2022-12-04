from aocd.models import Puzzle

puzzle = Puzzle(year=2015, day=1)


data = puzzle.input_data
print("a:", data.count("(") - data.count(")"))

floor = 0
for i, move in enumerate(data):
    floor += 1 if (move == "(") else -1
    if floor == -1:
        break
print("b:", i + 1)
