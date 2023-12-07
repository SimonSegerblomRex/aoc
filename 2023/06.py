import numpy as np
from aocd.models import Puzzle

YEAR = 2023
DAY = 6

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a_orig(times, distances):
    number_of_ways = []
    for time_record, distance_record in zip(times, distances):
        distance_traveled = []
        for time_holding in range(1, time_record):
            distance_traveled.append((time_record - time_holding) * time_holding)
        distance_traveled = np.array(distance_traveled)
        number_of_ways.append(
            len(distance_traveled[distance_traveled > distance_record])
        )
    return np.prod(number_of_ways)


from sympy import Symbol, reduce_inequalities
from sympy.core.relational import Relational
from sympy.parsing.sympy_parser import parse_expr
from sympy.solvers import solve


def a_sympy(times, distances):
    number_of_ways = []
    # (t_r - t) * t >= d, t > 0
    t = Symbol("t")
    for t_r, d_r in zip(times, distances):
        expr = parse_expr(f"({t_r} - t) * t > {d_r}, t > 0")
        eq = reduce_inequalities(expr, t)
        relations = [
            (i.lhs, i.rel_op, i.rhs)
            for i in [i.canonical for i in eq.atoms(Relational)]
        ]
        relations = sorted(relations, key=lambda x: float(x[2]))
        for relation in relations:
            if relation[1] == ">":
                left_limit = int(relation[2] + 1)
            elif relation[1] == ">=":
                left_limit = int(relation[2] + 0.5)
            elif relation[1] == "<":
                if relation[2].is_Integer:
                    right_limit = relation[2] - 1
                else:
                    right_limit = int(relation[2])
            elif relation[1] == "<=":
                right_limit = int(relation[2])
            else:
                raise ValueError("...")
        number_of_ways.append(right_limit - left_limit + 1)
    return np.prod(number_of_ways)


def a(times, distances):
    # (t_r - t) * t >= d, t > 0
    # => -sqrt(t_tr^2 / 4 - d) + t_r / 2 < t < sqrt(t_tr^2 / 4 - d) + t_r / 2
    number_of_ways = []
    for t_r, d_r in zip(times, distances):
        s = np.sqrt(t_r**2 / 4 - d_r)
        left_limit = int(-s + t_r / 2 + 1)
        right_limit = int(s + t_r / 2 + 1 - 1e-10) - 1
        number_of_ways.append(right_limit - left_limit + 1)
    return np.prod(number_of_ways)


example_answer = a(times=[7, 15, 30], distances=[9, 40, 200])
assert example_answer == 288
answer = a(times=[51, 69, 98, 78], distances=[377, 1171, 1224, 1505])
print("a:", answer)
assert answer == 131376


# Part b
example_answer = a(times=[71530], distances=[940200])
assert example_answer == 71503
answer = a(times=[51699878], distances=[377117112241505])
print("b:", answer)
assert answer == 34123437
