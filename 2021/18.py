import ast
from dataclasses import dataclass
from itertools import permutations

import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 18

puzzle = Puzzle(year=YEAR, day=DAY)


class SnailFishNumber:
    def __init__(self, val, depth=0, parent=None):
        self.parent = parent
        self.depth = depth
        if isinstance(val, int):
            self.val = val
            self.left = None
            self.right = None
        elif isinstance(val, list):
            self.val = None
            left, right = val
            self.left = SnailFishNumber(left, depth + 1, self)
            self.right = SnailFishNumber(right, depth + 1, self)
        else:
            raise ValueError("Can't handle val...")

    def __repr__(self):
        if self.val is not None:
            return str(self.val)
        return f"[{self.left}, {self.right}]"

    def leaves(self):
        if self.val is not None:
            return [self]
        l = self.left.leaves()
        l += self.right.leaves()
        return l

    def reduce(self):
        l = self.leaves()
        for i, e in enumerate(l):
            if e.depth == 5:
                # explode
                assert l[i + 1].val is not None
                if i > 0:
                    l[i - 1].val += e.val
                try:
                    l[i + 2].val += l[i + 1].val
                except IndexError:
                    pass
                e.parent.left = None
                e.parent.right = None
                e.parent.val = 0
                self.reduce()
                return
        for e in l:
            if e.val >= 10:
                # split
                e.left = SnailFishNumber(e.val // 2, e.depth + 1, e)
                e.right = SnailFishNumber((e.val + 1) // 2, e.depth + 1, e)
                e.val = None
                self.reduce()
                return

    def magnitude(self):
        if self.val is not None:
            return self.val
        return 3 * self.left.magnitude() + 2 * self.right.magnitude()


# Part a
def a(data):
    prev = None
    for n in data.split():
        if prev is None:
            prev = n
            continue
        l1 = ast.literal_eval(str(prev))
        l2 = ast.literal_eval(str(n))
        prev = SnailFishNumber([l1, l2])
        prev.reduce()
    return prev.magnitude()


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 3359


# Part b
def b(data):
    high_score = 0
    for n1, n2 in permutations(data.split(), 2):
        l1 = ast.literal_eval(n1)
        l2 = ast.literal_eval(n2)
        prev = SnailFishNumber([l1, l2])
        prev.reduce()
        high_score = max(prev.magnitude(), high_score)
    return high_score


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
assert answer == 4616
