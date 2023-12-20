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


def send_pulse(modules, sender, receiver, pulse):
    #print(sender, pulse, receiver)
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
            elif typ == "&":
                modules[module[1:]]["hist"] = defaultdict(lambda: 0)

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
            modules_to_process.extend(send_pulse(modules, sender, receiver, pulse))
    return pulses[0] * pulses[1]


answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    breakpoint()


for example in puzzle.examples:
    if example.answer_b:
        example_answer = b(example.input_data)
        print(f"Example answer: {example_answer} (expecting: {example.answer_b})")
        assert str(example_answer) == example.answer_b
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
