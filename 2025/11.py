from aocd.models import Puzzle

YEAR = 2025
DAY = 11

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    devices = {}
    for line in data.splitlines():
        key, val = line.split(":")
        devices[key] = val.strip().split(" ")

    def find_path(start, goal):
        if start == goal:
            return 1
        c = 0
        for device in devices[start]:
            c += find_path(device, goal)
        return c

    return find_path("you", "out")


answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 506


# Part b
def b(data):
    devices = {}
    for line in data.splitlines():
        key, val = line.split(":")
        devices[key] = val.strip().split(" ")

    cache = {}
    def find_path(start, goal, goal0=False, goal1=False):
        if (start, goal, goal0, goal1) in cache:
            return cache[(start, goal, goal0, goal1)]
        if start == goal:
            if goal0 and goal1:
                return 1
            return 0
        c = 0
        for device in devices[start]:
            c += find_path(
                device,
                goal,
                goal0=goal0 or device == "fft",
                goal1=goal1 or device == "dac",
            )
        cache[(start, goal, goal0, goal1)] = c
        return c

    return find_path("svr", "out")


answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 385912350172800
