from collections import Counter, defaultdict

from aocd.models import Puzzle

YEAR = 2024
DAY = 11

puzzle = Puzzle(year=YEAR, day=DAY)


def blink_helper(stone):
    if stone == 0:
        return [1]
    stone_str = str(stone)
    nbr_digits = len(stone_str)
    if not nbr_digits % 2:
        return [
            int(stone_str[: nbr_digits // 2]),
            int(stone_str[nbr_digits // 2 :]),
        ]
    return [2024 * stone]


def blink(stones):
    out = defaultdict(int)
    for stone, count in stones.items():
        new_stones = blink_helper(stone)
        for new_stone in new_stones:
            out[new_stone] += count
    return out


# Part a
def a(data, blinks=25):
    stones = [int(n) for n in data.split()]
    stones = Counter(stones)
    for _ in range(blinks):
        stones = blink(stones)
    return sum(stones.values())


example = "125 17"
assert a(example) == 55312
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 185205

# Part b
answer = a(puzzle.input_data, blinks=75)
print("b:", answer)
assert answer == 221280540398419
