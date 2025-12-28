import numpy as np
from aocd.models import Puzzle
from scipy.optimize import linprog

YEAR = 2025
DAY = 10

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def update_state(state, action):
    new_state = list(state)
    for idx in action:
        new_state[idx] ^= True
    return tuple(new_state)


def find_fewest_presses(state, goal, actions):
    states = [state]
    visited = set()
    presses = 0
    while states:
        new_states = []
        presses += 1
        for state in states:
            for action in actions:
                new_state = update_state(state, action)
                if new_state == goal:
                    return presses
                if new_state in visited:
                    continue
                visited.add(new_state)
                new_states.append(new_state)
        states = new_states


def a(data):
    goals = []
    all_actions = []
    for line in data.splitlines():
        goal_txt, *actions, _ = line.split(" ")
        goals.append(tuple(c == "#" for c in goal_txt.strip("[]")))
        all_actions.append(
            tuple(tuple(map(int, action.strip("()").split(","))) for action in actions)
        )
    s = 0
    for goal, actions in zip(goals, all_actions):
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
    goals = []
    all_actions = []
    for line in data.splitlines():
        _, *actions, goal_txt = line.split(" ")
        goals.append(tuple(map(int, goal_txt.strip(r"{}").split(","))))
        all_actions.append(
            tuple(tuple(map(int, action.strip("()").split(","))) for action in actions)
        )
    s = 0
    for ii, (goal, actions) in enumerate(zip(goals, all_actions)):
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
