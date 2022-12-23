from itertools import chain

import numpy as np
from aocd.models import Puzzle

YEAR = 2022
DAY = 23

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def debug_print(elves):
    # Debug print
    x, y = zip(*elves)
    width = max(x) - min(x) + 1
    height = max(y) - min(y) + 1
    x = np.array(x) - min(x)
    y = np.array(y) - min(y)
    tmpcoords = list(zip(x, y))
    print("")
    for y in range(height):
        for x in range(width):
            if (x, y) in tmpcoords:
                print("#", end="")
            else:
                print(".", end="")
        print("")
    breakpoint()


def a(data, rounds):
    data = data.replace(".", "0")
    data = data.replace("#", "1")
    rows = [
        np.frombuffer(row.encode(), dtype=np.uint8) - ord("0")
        for row in data.splitlines()
    ]
    grid = np.vstack(rows)
    y_coords, x_coords = np.nonzero(grid)
    elves = dict.fromkeys(zip(x_coords, y_coords))

    directions_check_order = ["north", "south", "west", "east"]

    round = 1
    while round <= rounds:
        coords = set(elves)
        # debug_print(elves)
        for x, y in elves:
            to_check = {
                "north": [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1)],
                "south": [(x - 1, y + 1), (x, y + 1), (x + 1, y + 1)],
                "west": [(x - 1, y + 1), (x - 1, y), (x - 1, y - 1)],
                "east": [(x + 1, y + 1), (x + 1, y), (x + 1, y - 1)],
            }
            for direction in directions_check_order:
                coords_in_direction = to_check[direction]
                if not coords.intersection(chain.from_iterable(to_check.values())):
                    break
                if not coords.intersection(coords_in_direction):
                    elves[(x, y)] = coords_in_direction[1]
                    break
        destinations = set()
        duplicate_destinations = {
            d for d in elves.values() if d in destinations or destinations.add(d)
        }
        duplicate_destinations.add(None)
        staying = {c: d for c, d in elves.items() if d in duplicate_destinations}
        if len(staying) == len(elves):
            # Answer for part b
            return round
        elves = dict.fromkeys((*(destinations - duplicate_destinations), *staying))
        directions_check_order.append(directions_check_order.pop(0))
        round += 1

    x, y = zip(*elves)
    width = max(x) - min(x) + 1
    height = max(y) - min(y) + 1
    # Answer for part a
    return width * height - len(elves)


example_answer = a(puzzle.example_data, rounds=10)
print(example_answer)
assert example_answer == 110
answer = a(puzzle.input_data, rounds=10)
print("a:", answer)
assert answer == 4075


# Part b
example_answer = a(puzzle.example_data, np.inf)
print(example_answer)
assert example_answer == 20
answer = a(puzzle.input_data, np.inf)
print("b:", answer)
assert answer == 950
