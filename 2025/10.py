import queue
import re
from collections import defaultdict
from functools import cache

import numpy as np
from aocd.models import Puzzle
from scipy.optimize import linprog

YEAR = 2025
DAY = 10

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
@cache
def update_state(state, action):
    new_state = list(state)
    for idx in action:
        new_state[idx] ^= True
    return tuple(new_state)


def find_fewest_presses(start, goal, actions, b_check=False):
    def h(next_state):
        return 0

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
    all_actions = tuple(
        tuple(tuple(map(int, e.strip("()").split(","))) for e in action.split())
        for action in actions
    )
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
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 434


# Part b
def b(data):
    goals = re.findall(r"{.+\}", data)
    actions = re.findall(r"\](.+){", data)
    all_goals = [tuple(map(int, goal.strip(r"{}").split(","))) for goal in goals]
    all_actions = tuple(
        tuple(tuple(map(int, e.strip("()").split(","))) for e in action.split())
        for action in actions
    )

    s = 0
    for ii, (goal, actions) in enumerate(zip(all_goals, all_actions)):
        A = np.zeros((len(goal), len(actions)), dtype=int)
        for j, action in enumerate(actions):
            for i in action:
                A[i, j] = 1
        bounds = [(0, max(goal))] * len(actions)
        c = [1] * len(actions)
        res = linprog(
            c,
            A_eq=A,
            b_eq=goal,
            bounds=bounds,
            integrality=1,
            options={"presolve": False},
        )
        s += int(res.fun)
    return s


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
assert answer == 15132
