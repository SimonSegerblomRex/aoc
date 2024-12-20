from aocd.models import Puzzle

YEAR = 2024
DAY = 20

puzzle = Puzzle(year=YEAR, day=DAY)


def manhattan(x, y):
    return int(abs(x.real - y.real) + abs(x.imag - y.imag))


def neighbours(p, dist):
    neighs = []
    for i in range(-dist, dist + 1):
        for j in range(-dist, dist + 1):
            if abs(i) + abs(j) > dist:
                continue
            n = p + j + i * 1j
            if n != p:
                neighs.append(n)
    return set(neighs)


# Part a
def a(data, cheat_dist=2):
    track = set()
    for i, line in enumerate(data.splitlines()):
        for j, c in enumerate(line):
            if c == "#":
                continue
            elif c == "S":
                start = j + i * 1j
            elif c == "E":
                goal = j + i * 1j
            track.add(j + i * 1j)
    pos = start
    path = [start]
    while pos != goal:
        neigh = neighbours(pos, 1)
        neigh &= track
        neigh -= set(path)
        neigh = neigh.pop()
        path.append(neigh)
        pos = neigh
    steps_from_start = dict(zip(path, range(len(path))))
    steps_to_goal = dict(zip(path, range(len(path) - 1, -1, -1)))
    shortest = steps_to_goal[start]
    c = 0
    for pos in path:
        neigh = set(neighbours(pos, cheat_dist))
        neigh &= set(path)
        for n in neigh:
            if (
                steps_from_start[pos] + manhattan(pos, n) + steps_to_goal[n]
                <= shortest - 100
            ):
                c += 1
    return c


answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 1409


# Part b
answer = a(puzzle.input_data, 20)
print("b:", answer)
assert answer == 1012821
