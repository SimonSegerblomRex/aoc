import datetime
import re
from collections import defaultdict

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def send_pulse_old(modules, sender, receiver, pulse):
    print(sender, receiver, pulse)
    pulses = [0, 0]
    pulses[pulse] += 1
    m = modules[receiver]
    typ = m["type"]
    if typ == "broadcaster":
        pass
    elif typ == "%":
        if pulse:
            return pulses
        m["state"] = 1 if (m["state"] == pulse) else 0
        pulse = m["state"]
    elif typ == "&":
        m["hist"][sender] = pulse
        if sum(m["hist"].values()) == len(m["hist"]):
            pulse = 0
        else:
            pulse = 1
    for d in m["dest"]:
        low, high = send_pulse(modules, receiver, d, pulse)
        pulses[0] += low
        pulses[1] += high
    return pulses


def send_pulse(modules, sender, receiver, pulse, i):
    #print(sender, pulse, receiver)
    if receiver == "rx" and pulse == 0:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return [8]
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
    """
    pulses = [0, 0]
    for _ in range(1000):
        low, high = send_pulse(modules, "button", "broadcaster", 0)
        pulses[0] += 1
        pulses[0] += low
        pulses[1] += high
        breakpoint()
    """
    pulses = [0, 0]
    curr_pulse = 0
    modules_to_process = [("button", "broadcaster", 0)]
    for _ in range(1000):
        modules_to_process = [("button", "broadcaster", 0)]
        while modules_to_process:
            sender, receiver, pulse = modules_to_process.pop(0)
            pulses[pulse] += 1
            modules_to_process.extend(send_pulse(modules, sender, receiver, pulse, 0))
    print(pulses)
    return pulses[0] * pulses[1]


# example = """broadcaster -> a
# %a -> inv, con
# &inv -> b
# %b -> con
# &con -> output"""
# example_answer = a(example)
# print(f"Example answer: {example_answer} (expecting: {11687500})")
# assert example_answer == 11687500


answer = a(puzzle.input_data)
print("a:", answer)
assert answer < 850401079
#puzzle.answer_a = answer


# Part b
def b(data):
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
    """
    pulses = [0, 0]
    for _ in range(1000):
        low, high = send_pulse(modules, "button", "broadcaster", 0)
        pulses[0] += 1
        pulses[0] += low
        pulses[1] += high
        breakpoint()
    """
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
        """
        for m in modules:
            if modules[m]["type"] == "&":
                for tmp2 in modules[m]["period"]:
                    if sum(modules[m]["hist"].values()) != len(modules[m]["hist"]):
                        modules[m]["period"][tmp2].append(0)
                        breakpoint()
        """
        if i > 100000:
            break
    lists = []
    for m in modules:
        if modules[m]["type"] == "&" and "bn" in modules[m]["dest"]:
            lists.append(modules[m]["period"])
    return np.prod([list(d.values())[0][1] for d in lists])
    return 3797*3881*4003*3823



answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
