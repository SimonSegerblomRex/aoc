import ast
import re

import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 18

puzzle = Puzzle(year=YEAR, day=DAY)


nodes

class SnailFishNumber:

    def __init__(self, list_, parent=None, depth=1):
        self.depth = depth
        self.parent = parent
        left, right = list_
        if isinstance(left, list):
            self.left = SnailFishNumber(left, self)
        else:
            self.left = left
        if isinstance(right, list):
            self.right = SnailFishNumber(right, self)
        else:
            self.right = right
        if parent is not None:
            parent.depth += 1
        while self.depth == 4:  # bara if parent is None?
            # Need to explode!
            self.explode()
            breakpoint()
            # while split criteria...

    def __add__(self, other):
        new = SnailFishNumber([self, other], None, max(self.depth, other.depth) + 1)
        self.parent = new
        other.parent = new
        return new

    def __repr__(self):
        return f"[{self.left}, {self.right}]"

    def _update_depth(self):
        if isinstance(SnailFishNumber, self.left):
            self.left._update_depth()
        if isinstance(SnailFishNumber, self.right):
            self.left._update_depth()
        self.depth += 1

# Part a
def a(data):
    #sn = ast.literal_eval("[[[[[9,8],1],2],3],4]")
    sn1 = SnailFishNumber([[[[4,3],4],4],[7,[[8,4],9]]])
    sn2 = SnailFishNumber([1,1])
    tmp = sn1 + sn2
    breakpoint()


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
#puzzle.answer_a = answer


# Part b
def b(data):
    exit()
