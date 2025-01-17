from aocd.models import Puzzle

YEAR = 2024
DAY = 6

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    obstacles = set()
    for i, line in enumerate(data.splitlines()):
        width = len(line)
        for j, c in enumerate(line):
            if c == "#":
                obstacles.add(j + i*1j)
            if c == "^":
                pos = j + i*1j
    height = i + 1
    dir = -1j
    c = 0
    visited = set()
    while (0 <= pos.real < width) and (0 <= pos.imag < height):
        visited.add(pos)
        if pos + dir in obstacles:
            dir *= 1j
            continue
        pos += dir
    return len(visited)


example = """....#.....
.........#
..........
..#.......
.......#..
..........
.#..^.....
........#.
#.........
......#..."""
answer = a(example)
assert answer == 41
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = 5444


# Part b
def b(data):
    obstacles = set()
    for i, line in enumerate(data.splitlines()):
        width = len(line)
        for j, c in enumerate(line):
            if c == "#":
                obstacles.add(j + i*1j)
            if c == "^":
                pos = j + i*1j
    height = i + 1
    orig_pos = pos

    dir = -1j
    visited = set()
    while (0 <= pos.real < width) and (0 <= pos.imag < height):
        visited.add(pos)
        if pos + dir in obstacles:
            dir *= 1j
            continue
        pos += dir

    c = 0
    for tmp in visited:
        obstacles.add(tmp)
        prev_state = set()
        pos = orig_pos
        dir = -1j
        while (0 <= pos.real < width) and (0 <= pos.imag < height):
            if (pos, dir) in prev_state:
                c += 1
                break
            prev_state.add((pos, dir))
            if pos + dir in obstacles:
                dir *= 1j
                continue
            pos += dir
        obstacles.remove(tmp)
    return c


answer = b(example)
print(answer)
assert answer == 6
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 1946