import numpy as np
from aocd.models import Puzzle
from sympy import Symbol
from sympy.parsing.sympy_parser import parse_expr
from sympy.solvers import solve

YEAR = 2022
DAY = 21

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
d = {}


def a(data):
    for line in data.replace(":", "").splitlines():
        c = line.split(" ")
        try:
            value = int(c[1])
            assert len(c) == 2
            d[c[0]] = eval(f"lambda *_: {c[1]}")
        except ValueError:
            d[c[0]] = eval(f"lambda *_: d['{c[1]}']() {c[2]} d['{c[3]}']()")
    return int(d["root"]())


example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 152
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 121868120894282


# Part b
def b(data):
    d = dict(line.split(": ") for line in data.splitlines())
    d["humn"] = "x"

    def get_new_val(v, root=False):
        if v == "x":
            return v
        c = v.split(" ")
        try:
            int(v)
            return v
        except ValueError as e:
            return f"({get_new_val(d[c[0]])}) {c[1] if not root else '='} ({get_new_val(d[c[2]])})"

    d = {k: get_new_val(v, root=k == "root") for k, v in d.items()}
    expr1, expr2 = d["root"].split(" = ")
    x = Symbol("x")
    return solve(parse_expr(expr1) - parse_expr(expr2), x)[0]


example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 301
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 3582317956029
