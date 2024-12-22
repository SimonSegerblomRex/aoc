from itertools import product

from aocd.models import Puzzle

YEAR = 2024
DAY = 22

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    s = 0
    for n in data.split():
        for _ in range(2000):
            n = int(n)
            n ^= 64 * n
            n %= 16777216
            n ^= n // 32
            n %= 16777216
            n ^= 2048 * n
            n %= 16777216
        s += n
    return s



example = """1
10
100
2024"""
answer = a(example)
assert answer == 37327623
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 12664695565


# Part b
def b(data):
    prices = []
    diff = []
    for n in data.split():
        n = int(n)
        tmp_p = [n % 10]
        tmp_d = [0]
        for _ in range(2000):
            n = int(n)
            n ^= 64 * n
            n %= 16777216
            n ^= n // 32
            n %= 16777216
            n ^= 2048 * n
            n %= 16777216
            tmp_d.append(n % 10 - tmp_p[-1])
            tmp_p.append(n % 10)
        prices.append(tmp_p)
        diff.append(tmp_d)
    seqs = []
    for p, d in zip(prices, diff):
        seq = set()
        for i in range(1, len(p) - 3):
            if d[i + 3] > 0:
                seq.add(tuple(d[i:i + 4]))
        seqs.append(seq)
    bananas = []
    for n, seq in enumerate(set.union(*seqs)):
        s = 0
        seq_list = list(seq)
        for i, (p, d) in enumerate(zip(prices, diff)):
            if seq not in seqs[i]:
                continue
            for idx in range(1, len(p) - 3):
                if d[idx:idx + 4] == seq_list:
                    s += p[idx + 3]
                    break
        bananas.append(s)
    return max(bananas)


example = """1
2
3
2024"""
answer = b(example)
print(answer)
assert answer == 23
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 1444
