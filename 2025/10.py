import datetime
import re
import queue
from collections import defaultdict
from functools import cache

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = 10

puzzle = Puzzle(year=YEAR, day=DAY)


import sys
sys.setrecursionlimit(5000)

# Part a
@cache
def update_state(state, action):
    new_state = list(state)
    for idx in action:
        new_state[idx] ^= True
    return tuple(new_state)


def find_fewest_presses(start, goal, actions, b_check=False):
    def h(next_state):
        return 1#sum(g - s for g, s in zip(goal, next_state))

    open_set = queue.PriorityQueue()
    # (f_score, state, prev_action)
    open_set.put((0, start, None))

    g_score = defaultdict(lambda: 1 << 30)
    g_score[(start, None)] = 0

    f_score = defaultdict(lambda: 1 << 30)
    f_score[(start, None)] = h(start)

    while not open_set.empty():
        _, curr_state, prev_action = open_set.get()
        if curr_state == goal:
            return int(g_score[(curr_state, prev_action)])

        for action in actions:
            if not b_check and action == prev_action:
                continue
            next_state = update_state(curr_state, action)
            if b_check and any(s > g for s, g in zip(next_state, goal)):
                continue
            tentative_g_score = g_score[(curr_state, prev_action)] + 1
            if tentative_g_score < g_score[(next_state, action)]:
                g_score[(next_state, action)] = tentative_g_score
                f_score[(next_state, action)] = tentative_g_score + h(next_state)
                open_set.put(
                    (
                        f_score[(next_state, action)],
                        next_state,
                        action,
                    )
                )


def a(data):
    goals = re.findall(r"\[.+\]", data)
    actions = re.findall(r"\](.+){", data)
    all_goals = [tuple(c == "#" for c in goal.strip("[]")) for goal in goals]
    all_actions = tuple(tuple(tuple(map(int, e.strip("()").split(","))) for e in action.split()) for action in actions)
    s = 0
    for goal, actions in zip(all_goals, all_actions):
        state = tuple([False] * len(goal))
        s += find_fewest_presses(state, goal, actions)
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
#answer = a(puzzle.input_data)
#print("a:", answer)
#puzzle.answer_a = answer


# Part b
def update_state(state, action):
    new_state = list(state)
    for idx in action:
        new_state[idx] += 1
    return tuple(new_state)


def update_history(state, idx):
    new_state = list(state)
    new_state[idx] += 1
    return tuple(new_state)


def find_best_action(state, goal, actions):
    # Look for indeces that need many keypresses

    breakpoint()

best_solution = None

def b(data):
    goals = re.findall(r"{.+\}", data)
    actions = re.findall(r"\](.+){", data)
    all_goals = [tuple(map(int, goal.strip(r"{}").split(","))) for goal in goals]
    all_actions = tuple(tuple(tuple(map(int, e.strip("()").split(","))) for e in action.split()) for action in actions)

    s = 0
    from sympy import solve_linear_system, solve, symbols, solveset, linear_eq_to_matrix, linsolve, Eq, Ge, Le
    from sympy.core.numbers import Integer
    from sympy.solvers.simplex import lpmin
    from sympy import ceiling
    for ii, (goal, actions) in enumerate(zip(all_goals, all_actions)):
        if 0:
            global best_solution
            best_solution = defaultdict(lambda: 1 << 30)
            def find_solutions(start, goal, actions, steps, upper):
            #def find_solutions(start, goal, actions, history, upper):
                global best_solution
                if steps < best_solution[start]:
                    best_solution[start] = steps
                else:
                    return best_solution[start]
                #if any(s > g for s, g in zip(history, upper)):
                #    return
                if start == goal:
                    return steps
                # find best action to take
                wanted = set(i for i, (g, s) in enumerate(zip(goal, start)) if g > s)
                action_score = []
                for action in actions:
                    dist = sum(goal[i]-start[i] for i in action)
                    #action_score.append(dist)
                    action_score.append((len(set(action) & wanted), dist))
                tmp = []
                for _, i, action in sorted(zip(action_score, range(len(actions)), actions), reverse=True):
                    tmp.append(find_solutions(update_state(start, action), goal, actions, steps + 1, upper))
                return min(tmp)

            actions = tuple(sorted(actions, key=lambda a: len(a), reverse=True))

            upper_bounds = []
            for action in actions:
                upper_bounds.append(min(goal[idx] for idx in action))

            #find_solutions(tuple([0]*len(goal)), goal, actions, tuple([0]*len(actions)), tuple(upper_bounds))
            find_solutions(tuple([0]*len(goal)), goal, actions, 0, tuple(upper_bounds))

            s += best_solution
            print(ii)
            continue

        upper_bounds = []
        for action in actions:
            upper_bounds.append(min(goal[idx] for idx in action))


        A = np.zeros((len(goal), len(actions)), dtype=int)
        for j, action in enumerate(actions):
            for i in action:
                A[i, j] = 1

        from scipy.optimize import linprog
        bounds = [(0, b) for b in upper_bounds]
        c = [1]*len(actions)
        res = linprog(c, A_eq=A, b_eq=goal, bounds=bounds, integrality=1)
        s += int(res.fun)
        if ii == 7:
            breakpoint()
        continue
        breakpoint()


        x_symbols = symbols(f"x:{len(actions)}", integer=True)#, positive=True)
        equations = []
        for row, g in zip(A, goal):
            equations.append(sum(row * x_symbols) - g)
        solution = solve(equations, *x_symbols)
        constr =[]

        constr = [Eq(k, v) for k, v in solution.items()]
        for i, x in enumerate(x_symbols):
            constr.append(Ge(x, 0))
            constr.append(Le(x, upper_bounds[i]))
        for k, v in solution.items():
            constr.append(Eq(k, v))

        tmp = lpmin(sum(x_symbols), constr)
        #s += int(tmp[0])
        print(ii, tmp[0], tmp)
        bb =  A @ np.array(list(tmp[1].values())).astype(int)
        s += int(tmp[0])
        if tuple(bb) != goal:
            s += find_fewest_presses(tuple(goal - bb), goal, actions, b_check=True)
        continue
        breakpoint()
        #tt = np.concatenate((A, np.expand_dims(goal, axis=1)), axis=1)
        #sol2 = solve_linear_system(tt, *x_symbols)
        #sol3 = linsolve(equations, *x_symbols)
        #breakpoint()
        #tmp = sum(solution.values())
        if x_to_set_to_zero := set(x_symbols) - set(solution.keys()):
            idd = []
            for x in x_to_set_to_zero:
                idd.append(list(x_symbols).index(x))
            A = A[:, [i for i in range(A.shape[1]) if i not in idd]]
            x_symbols = symbols(f"x:{A.shape[1]}", integer=True, positive=True)
            equations2 = []
            for row, g in zip(A, goal):
                equations2.append(sum(row * x_symbols) - g)
            solution2 = solve(equations2, *solution.keys())
            breakpoint()
        if isinstance(tmp, Integer):
            s += tmp
            print(solution)
        else:
            breakpoint()
        print(ii)
        continue

        A = Matrix(A)
        bb = Matrix(goal)
        s += sum
        breakpoint()
        #t = 0
        #state = tuple([0] * len(goal))
        #while True:
        #    # Don't go all the way...
        #    action = find_best_action(state, goal, actions)
        #    if not action:
        #        breakpoint()
        #        break
        #    state = update_state(state, action)
        #    t += 1
        #t += find_fewest_presses(state, goal, actions, b_check=True) or 0
        #s += t
        print(i)
    return s


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
assert answer > 14954
assert answer < 15244
assert answer != 15102
assert answer != 15131
print("b:", answer)
puzzle.answer_b = answer
