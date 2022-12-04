import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 3

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    bits = len(data.split("\n")[0])
    print(bits)
    data = np.vstack(
        [np.frombuffer(n.encode(), dtype=np.uint8) - ord("0") for n in data.split("\n")]
    )
    data = data[:, -bits:]
    nbr_ones = data.sum(axis=0)
    gamma = (nbr_ones > data.shape[0] // 2).astype(np.uint8)
    epsilon = 1 - gamma
    gamma = int("".join(str(b) for b in gamma.tolist()), 2)
    epsilon = int("".join(str(b) for b in epsilon.tolist()), 2)
    return gamma * epsilon


assert a(puzzle.example_data) == 198
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 3277364


# Part b
def b(data):
    bits = len(data.split("\n")[0])
    print(bits)
    data = np.vstack(
        [np.frombuffer(n.encode(), dtype=np.uint8) - ord("0") for n in data.split("\n")]
    )
    oxygen = data[:, -bits:]
    # Oxygen
    for c in range(oxygen.shape[1]):
        counts = np.bincount(oxygen[:, c])
        if counts[0] == counts[1]:
            zero_or_one = 1
        else:
            zero_or_one = np.argmax(counts)
        oxygen = oxygen[oxygen[:, c] == zero_or_one, :]
        if oxygen.shape[0] == 1:
            break
    oxygen = oxygen[0]
    # CO2
    co2 = data[:, -bits:]
    for c in range(co2.shape[1]):
        counts = np.bincount(co2[:, c])
        if counts[0] == counts[1]:
            zero_or_one = 0
        else:
            zero_or_one = np.argmin(counts)
        co2 = co2[co2[:, c] == zero_or_one, :]
        if co2.shape[0] == 1:
            break
    co2 = co2[0]
    oxygen = int("".join(str(b) for b in oxygen.tolist()), 2)
    co2 = int("".join(str(b) for b in co2.tolist()), 2)
    return oxygen * co2

assert  b(puzzle.example_data) == 230
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 5736383
