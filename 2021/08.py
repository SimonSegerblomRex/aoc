import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 8

puzzle = Puzzle(year=YEAR, day=DAY)

puzzle.example_data_real = """be cfbegad cbdgef fgaecd cgeb fdcge agebfd fecdb fabcd edb | fdgacbe cefdb cefbgd gcbe
edbfga begcd cbg gc gcadebf fbgde acbgfd abcde gfcbed gfec | fcgedb cgb dgebacf gc
fgaebd cg bdaec gdafb agbcfd gdcbef bgcad gfac gcb cdgabef | cg cg fdcagb cbg
fbegcd cbd adcefb dageb afcb bc aefdc ecdab fgdeca fcdbega | efabcd cedba gadfec cb
aecbfdg fbg gf bafeg dbefa fcge gcbea fcaegb dgceab fcbdga | gecf egdcabf bgf bfgea
fgeab ca afcebg bdacfeg cfaedg gcfdb baec bfadeg bafgc acf | gebdcfa ecba ca fadegcb
dbcfg fgd bdegcaf fgec aegbdf ecdfab fbedc dacgb gdcebf gf | cefg dcbef fcge gbcadfe
bdfegc cbegaf gecbf dfcage bdacg ed bedf ced adcbefg gebcd | ed bcgafe cdgba cbgef
egadfb cdbfeg cegd fecab cgb gbdefca cg fgcdab egfdb bfceg | gbdfcae bgc cg cgb
gcafb gcf dcaebfg ecagb gf abcdeg gaef cafbge fdbac fegbdc | fgae cfgab fg bagce
"""

# Part a
def a(data):
    outputs = []
    for line in data.splitlines():
        outputs += line.split("|")[-1].strip(" ").split(" ")
    nbr_segments = np.array([len(digit) for digit in outputs])
    counts = np.bincount(nbr_segments, minlength=7)
    return counts[[2, 4, 3, 7]].sum()


example_answer = a(puzzle.example_data_real)
print(example_answer)
assert example_answer == 26
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 470


# Part b
def b(data):
    total = 0
    for line in data.splitlines():
        signals, outputs = line.split("|")
        signals = np.array(signals.strip(" ").split(" "))
        nbr_segments = np.array([len(digit) for digit in signals])
        digits = {}
        digits[1] = signals[nbr_segments == 2][0]
        digits[4] = signals[nbr_segments == 4][0]
        digits[7] = signals[nbr_segments == 3][0]
        digits[8] = signals[nbr_segments == 7][0]
        signals_6 = signals[nbr_segments == 6].tolist()
        for s in signals_6:
            tmp = [c in s for c in digits[4]]
            if np.all(tmp):
                break
        digits[9] = s
        signals_6.remove(s)
        for s in signals_6:
            tmp = [c in s for c in digits[1]]
            if np.all(tmp):
                break
        digits[0] = s
        signals_6.remove(s)
        digits[6] = signals_6[0]
        signals_5 = signals[nbr_segments == 5].tolist()
        for s in signals_5:
            tmp = [c in s for c in digits[1]]
            if np.all(tmp):
                break
        digits[3] = s
        signals_5.remove(s)
        for s in signals_5:
            tmp = [c in digits[6] for c in s]
            if np.all(tmp):
                break
        digits[5] = s
        signals_5.remove(s)
        digits[2] = signals_5[0]
        signals_to_digits = {"".join(sorted(v)): k for k, v in digits.items()}
        outputs = ["".join(sorted(output)) for output in outputs.strip(" ").split(" ")]
        number = int("".join(str(signals_to_digits[output]) for output in outputs))
        total += number
    return total


example_answer = b(puzzle.example_data_real)
print(example_answer)
assert example_answer == 61229
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 989396
