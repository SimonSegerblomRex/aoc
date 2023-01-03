import numpy as np
from aocd.models import Puzzle

YEAR = 2022
DAY = 25

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def dec2SNAFU(x):
    l = np.base_repr(x, base=5)
    l = [0, *list(map(int, l))]
    o = []
    r = 0
    for i, v in zip(range(len(l) - 1, -1, -1), l[::-1]):
        v += r
        if v > 2:
            if v == 3:
                o.append("=")
            elif v == 4:
                o.append("-")
            elif v == 5:
                o.append("0")
                rem = 1
            else:
                print("Shouldn't end up here")
                print(v)
                breakpoint()
            r = 1
        else:
            o.append(v)
            r = 0
    t = "".join(map(str, o))[::-1]
    return t[1:] if (t[0] == "0") else t


def SNAFU2dec(x):
    l = list(x)
    o = 0
    for i, v in zip(range(len(l) - 1, -1, -1), l):
        if v == "=":
            v = -2
        elif v == "-":
            v = -1
        else:
            v = int(v)
        o += v * 5**i
    return o


def a(data):
    return dec2SNAFU(sum(map(SNAFU2dec, data.splitlines())))


assert dec2SNAFU(12345) == "1-0---0"
assert dec2SNAFU(314159265) == "1121-1110-1=0"

assert SNAFU2dec("1=") == 3
assert SNAFU2dec("2=0=") == 198
assert SNAFU2dec("1=-0-2") == 1747

example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == "2=-1=0"
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == "2-00=12=21-0=01--000"
