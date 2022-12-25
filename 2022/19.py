import re
from dataclasses import dataclass
from functools import partial

import numpy as np
from aocd.models import Puzzle

YEAR = 2022
DAY = 19

puzzle = Puzzle(year=YEAR, day=DAY)

EXAMPLE_DATA = """Blueprint 1:
  Each ore robot costs 4 ore.
  Each clay robot costs 2 ore.
  Each obsidian robot costs 3 ore and 14 clay.
  Each geode robot costs 2 ore and 7 obsidian.

Blueprint 2:
  Each ore robot costs 2 ore.
  Each clay robot costs 3 ore.
  Each obsidian robot costs 3 ore and 8 clay.
  Each geode robot costs 3 ore and 12 obsidian."""

PATTERN = r"Blueprint (\d+):\s+Each ore robot costs (\d+) ore.\s+Each clay robot costs (\d+) ore.\s+Each obsidian robot costs (\d+) ore and (\d+) clay.\s+Each geode robot costs (\d+) ore and (\d+) obsidian."

Cost = partial(np.array, dtype=int)

@dataclass
class Blueprint:
    nbr: int
    costs: list


# Part a
def a(data):
    data = re.findall(PATTERN, data)
    blueprints = [
        Blueprint(
            nbr=int(e[0]),
            costs=np.array([
                Cost((e[1], 0, 0, 0)),
                Cost((e[2],0 ,0, 0)),
                Cost((e[3], e[4], 0, 0)),
                Cost((e[5], 0, e[6], 0)),
            ]),
        )
        for e in data
    ]

    s = 0
    for blueprint in blueprints:
        best_so_far = [0]
        debug_options = [1, 1, 1, 2, 1, 2, 3, 3]
        def find_paths(blueprint, robots, stock, time_remaining):
            next_robot_options = [i for i, c in enumerate(blueprint.costs) if robots[np.flatnonzero(c)].all()]
            already_enough_robots = np.flatnonzero((robots >= blueprint.costs.max(axis=0))[:3])
            next_robot_options = list(set(next_robot_options) - set(already_enough_robots))
            if robots[-1] * time_remaining + time_remaining * (time_remaining + 1) / 2 + stock[-1] < best_so_far[0]:
                next_robot_options = []
            #for next_robot in [] if not debug_options else [debug_options.pop(0)]:
            for next_robot in next_robot_options[::-1]:
                if ((blueprint.costs[next_robot] - stock) <= 0).all():
                    time_required = 1
                else:
                    time_required = int(np.ceil(np.nanmax((blueprint.costs[next_robot] - stock) / robots))) + 1
                    if time_required <= 0:
                        raise ValueError
                    #breakpoint()
                next_time_remaining = time_remaining - time_required
                if next_time_remaining < 0:
                    continue
                next_stock = stock + time_required * robots - blueprint.costs[next_robot]
                next_robots = robots.copy()
                next_robots[next_robot] += 1
                if 0:
                    print("Current:", "robots:", robots, "stock:", stock, "time_remaining:", time_remaining)
                    print("Next:", "robots:", next_robots, "stock:", next_stock, "time_remaining:", next_time_remaining)
                yield from find_paths(
                    blueprint, next_robots, next_stock, next_time_remaining
                )
            stock += robots * time_remaining
            best_so_far[0] = max(best_so_far[0], stock[-1])
            yield stock[-1]

        robots = np.array((1, 0, 0, 0))
        stock = np.array((0, 0, 0, 0))
        s += blueprint.nbr * max(find_paths(blueprint, robots, stock, 24))
    return int(s)


"""
example_answer = a(EXAMPLE_DATA)
print(example_answer)
assert example_answer == 33
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 1466
"""


# Part b
def b(data):
    data = re.findall(PATTERN, data)
    blueprints = [
        Blueprint(
            nbr=int(e[0]),
            costs=np.array([
                Cost((e[1], 0, 0, 0)),
                Cost((e[2],0 ,0, 0)),
                Cost((e[3], e[4], 0, 0)),
                Cost((e[5], 0, e[6], 0)),
            ]),
        )
        for e in data
    ]

    s = []
    for blueprint in blueprints[:3]:
        best_so_far = [0]
        debug_options = [1, 1, 1, 2, 1, 2, 3, 3]
        def find_paths(blueprint, robots, stock, time_remaining):
            next_robot_options = [i for i, c in enumerate(blueprint.costs) if robots[np.flatnonzero(c)].all()]
            already_enough_robots = np.flatnonzero((robots >= blueprint.costs.max(axis=0))[:3])
            next_robot_options = list(set(next_robot_options) - set(already_enough_robots))
            if robots[-1] * time_remaining + time_remaining * (time_remaining + 1) / 2 + stock[-1] < best_so_far[0]:
                next_robot_options = []
            #for next_robot in [] if not debug_options else [debug_options.pop(0)]:
            for next_robot in next_robot_options[::-1]:
                if ((blueprint.costs[next_robot] - stock) <= 0).all():
                    time_required = 1
                else:
                    time_required = int(np.ceil(np.nanmax((blueprint.costs[next_robot] - stock) / robots))) + 1
                    if time_required <= 0:
                        raise ValueError
                    #breakpoint()
                next_time_remaining = time_remaining - time_required
                if next_time_remaining < 0:
                    continue
                next_stock = stock + time_required * robots - blueprint.costs[next_robot]
                next_robots = robots.copy()
                next_robots[next_robot] += 1
                if 0:
                    print("Current:", "robots:", robots, "stock:", stock, "time_remaining:", time_remaining)
                    print("Next:", "robots:", next_robots, "stock:", next_stock, "time_remaining:", next_time_remaining)
                yield from find_paths(
                    blueprint, next_robots, next_stock, next_time_remaining
                )
            stock += robots * time_remaining
            best_so_far[0] = max(best_so_far[0], stock[-1])
            yield stock[-1]

        robots = np.array((1, 0, 0, 0))
        stock = np.array((0, 0, 0, 0))
        s += blueprint.nbr * max(find_paths(blueprint, robots, stock, 32))
    return int(np.prod(s))


answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
