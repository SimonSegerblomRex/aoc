import ast
from dataclasses import dataclass
from itertools import permutations

import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 18

puzzle = Puzzle(year=YEAR, day=DAY)


class SnailFishNumber:
    def __init__(self, val, parent=None, depth=0, i=None):
        self.parent = parent
        if parent is None:
            # root
            self.depth = 0
            self.leaves = []
        else:
            self.depth = parent.depth + 1
            self.leaves = parent.leaves
        if isinstance(val, int):
            # leaf
            self.val = val
            self.left = None
            self.right = None
            if i is None:
                self.leaves.append(self)
            else:
                self.leaves.insert(i, self)
        elif isinstance(val, list):
            self.val = None
            left, right = val
            self.left = SnailFishNumber(left, self)
            self.right = SnailFishNumber(right, self)
        else:
            raise ValueError("Can't handle val...")

    def __repr__(self):
        if self.val is not None:
            return str(self.val)
        return f"[{self.left}, {self.right}]"

    def explode(self):
        i = self.leaves.index(self.left)
        assert self.right in self.leaves
        if i > 0:
            self.leaves[i - 1].val += self.left.val
        try:
            self.leaves[i + 2].val += self.right.val
        except IndexError:
            pass
        self.leaves.pop(i)
        self.leaves.pop(i)
        self.leaves.insert(i, self)
        self.left = None
        self.right = None
        self.val = 0

    def split(self):
        i = self.leaves.index(self)
        self.right = SnailFishNumber((self.val + 1) // 2, self, i=i)
        self.left = SnailFishNumber(self.val // 2, self, i=i)
        self.val = None
        self.leaves.remove(self)

    def reduce(self):
        for e in self.leaves:
            if e.depth == 5:
                e.parent.explode()
                self.reduce()
                return
        for e in self.leaves:
            if e.val >= 10:
                e.split()
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
