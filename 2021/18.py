import ast
import re
from dataclasses import dataclass

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
        if self.left is None and self.right is None:
            return str(self.val)
        return f"[{self.left}, {self.right}]"

    def leaves(self, l=None):
        if l is None:
            l = []
        if self.left is None and self.right is None:
            l.append(self)
            return l
        l.extend(e for e in self.left.leaves(l) if e not in l)
        l.extend(e for e in self.right.leaves(l) if e not in l)
        return l

    def magnitude(self):
        if self.val is not None:
            return self.val
        return 3 * self.left.magnitude() + 2 * self.right.magnitude()


def reduce(n):
    l = n.leaves()
    for i, e in enumerate(l):
        if e.depth == 5:
            # explode
            assert(l[i + 1].val is not None)
            if i > 0:
                l[i - 1].val += e.val
            try:
                l[i + 2].val += l[i + 1].val
            except IndexError:
                pass
            e.parent.left = None
            e.parent.right = None
            e.parent.val = 0
            return True
    for i, e in enumerate(l):
        if e.val >= 10:
            # split
            e.left = SnailFishNumber(e.val // 2, e.depth + 1, e)
            e.right = SnailFishNumber((e.val + 1) // 2, e.depth + 1, e)
            e.val = None
            return True
    return False


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
        while reduce(prev):
            pass
    return prev.magnitude()


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    exit()
