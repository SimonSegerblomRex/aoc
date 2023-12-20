from aocd.models import Puzzle

YEAR = 2023
DAY = 20

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def parse_modules(data):
    modules = {}
    for line in data.splitlines():
        items = line.split(" -> ")
        module = items[0]
        items = items[1].split(",")
        items = [item.strip() for item in items]
        if module in ("broadcaster"):
            modules[module] = {
                "type": module,
                "dest": items,
            }
        else:
            typ = module[0]
            modules[module[1:]] = {
                "type": typ,
                "dest": items,
            }
            if typ == "%":
                modules[module[1:]]["state"] = 0
    for module in modules:
        if modules[module]["type"] == "&":
            modules[module]["hist"] = {}
            modules[module]["period"] = {}
            for m in modules:
                if module in modules[m]["dest"]:
                    modules[module]["hist"][m] = 0
                    modules[module]["period"][m] = [0]
    return modules


def send_pulse(modules, sender, receiver, pulse, i):
    if receiver not in modules:
        return []
    m = modules[receiver]
    typ = m["type"]
    if typ == "broadcaster":
        pass
    elif typ == "%":
        if pulse:
            return []
        m["state"] = 1 if (m["state"] == pulse) else 0
        pulse = m["state"]
    elif typ == "&":
        m["hist"][sender] = pulse
        if sum(m["hist"].values()) == len(m["hist"]):
            pulse = 0
        else:
            pulse = 1
            if len(m["period"][sender]) < 3:
                m["period"][sender].append(i)
    return [(receiver, dest, pulse) for dest in m["dest"]]


def a(data):
    modules = parse_modules(data)
    pulses = [0, 0]
    curr_pulse = 0
    modules_to_process = [("button", "broadcaster", 0)]
    for _ in range(1000):
        modules_to_process = [("button", "broadcaster", 0)]
        while modules_to_process:
            sender, receiver, pulse = modules_to_process.pop(0)
            pulses[pulse] += 1
            modules_to_process.extend(send_pulse(modules, sender, receiver, pulse, 0))
    return pulses[0] * pulses[1]


example = """broadcaster -> a
%a -> inv, con
&inv -> b
%b -> con
&con -> output"""
example_answer = a(example)
print(f"Example answer: {example_answer} (expecting: {11687500})")
assert example_answer == 11687500
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 825167435


# Part b
def b(data):
    modules = parse_modules(data)
    pulses = [0, 0]
    curr_pulse = 0
    modules_to_process = [("button", "broadcaster", 0)]
    i = 1
    while True:
        modules_to_process = [("button", "broadcaster", 0)]
        while modules_to_process:
            sender, receiver, pulse = modules_to_process.pop(0)
            pulses[pulse] += 1
            tmp = send_pulse(modules, sender, receiver, pulse, i)
            modules_to_process.extend(tmp)
        i += 1
        answer = 1
        for m in modules.values():
            if m["type"] == "&" and "bn" in m["dest"]:
                if len(list(m["period"].values())[0]) < 2:
                    answer = 0
                    break
                else:
                    answer *= list(m["period"].values())[0][1]
        if answer:
            return answer


answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 225514321828633
