from aocd.models import Puzzle

YEAR = 2023
DAY = 19

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    workflows_input, ratings_input = data.split("\n\n")
    workflows = {}
    for line in workflows_input.splitlines():
        name, rules = line.split("{")
        rules = rules.strip("}").split(",")
        workflows[name] = rules
    s = 0
    for line in ratings_input.splitlines():
        part_spec = {}
        for entry in line.strip("{}").split(","):
            category, value = entry.split("=")
            part_spec[category] = int(value)
        curr_workflow = "in"
        while True:
            rules = workflows[curr_workflow]
            for rule in rules:
                if "<" in rule:
                    c, v = rule.split("<")
                    v, dest = v.split(":")
                    if part_spec[c] < int(v):
                        curr_workflow = dest
                        break
                elif ">" in rule:
                    c, v = rule.split(">")
                    v, dest = v.split(":")
                    if part_spec[c] > int(v):
                        curr_workflow = dest
                        break
                else:
                    curr_workflow = rule
                    break
            if curr_workflow == "A":
                s += sum(part_spec.values())
                break
            if curr_workflow == "R":
                break
    return s


for example in puzzle.examples:
    if example.answer_a:
        example_answer = a(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_a})")
        assert str(example_answer) == example.answer_a
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 350678


# Part b
def score(min_limits, max_limits):
    s = 1
    for lower_limit, upper_limit in zip(min_limits.values(), max_limits.values()):
        s *= upper_limit - lower_limit + 1
    return s


def number_of_combinations(workflows, curr_workflow, min_limits, max_limits):
    s = 0
    while True:
        if curr_workflow == "R":
            return s
        if curr_workflow == "A":
            return s + score(min_limits, max_limits)
        rules = workflows[curr_workflow]
        for rule in rules:
            if "<" in rule:
                c, v = rule.split("<")
                v, dest = v.split(":")
                if min_limits[c] < int(v):
                    s += number_of_combinations(
                        workflows,
                        dest,
                        min_limits.copy(),
                        {**max_limits, c: int(v) - 1},
                    )
                    min_limits[c] = int(v)
            elif ">" in rule:
                c, v = rule.split(">")
                v, dest = v.split(":")
                if max_limits[c] > int(v):
                    s += number_of_combinations(
                        workflows,
                        dest,
                        {**min_limits, c: int(v) + 1},
                        max_limits.copy(),
                    )
                    max_limits[c] = int(v)
            else:
                curr_workflow = rule
                break


def b(data):
    workflows_input, _ = data.split("\n\n")
    workflows = {}
    for line in workflows_input.splitlines():
        name, rules = line.split("{")
        rules = rules.strip("}").split(",")
        workflows[name] = rules
    min_limits = {"x": 1, "m": 1, "a": 1, "s": 1}
    max_limits = {"x": 4000, "m": 4000, "a": 4000, "s": 4000}
    return number_of_combinations(workflows, "in", min_limits, max_limits)


for example in puzzle.examples:
    example_answer = b(example.input_data)
    print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
    assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 124831893423809
