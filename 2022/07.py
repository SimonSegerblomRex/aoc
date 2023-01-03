import datetime
import re

import numpy as np
from aocd.models import Puzzle

YEAR = datetime.datetime.today().year
DAY = datetime.datetime.today().day

puzzle = Puzzle(year=YEAR, day=DAY)

example = """$ cd /
$ ls
dir a
14848514 b.txt
8504156 c.dat
dir d
$ cd a
$ ls
dir e
29116 f
2557 g
62596 h.lst
$ cd e
$ ls
584 i
$ cd ..
$ cd ..
$ cd d
$ ls
4060174 j
8033020 d.log
5626152 d.ext
7214296 k
"""

# Part a
def a(data):
    directories = []
    files = []
    cwd = []
    lines = iter(data.splitlines())
    for line in lines:
        if line.startswith("$"):
            _, command, *arg = line.split(" ")
            if command == "cd":
                if arg[0] == "..":
                    cwd.pop()
                else:
                    directories.append("".join(cwd).replace("//", "/") + arg[0] + "/")
                    cwd.append(arg[0] + "/")
            elif command == "ls":
                pass
        elif line.startswith("dir "):
            pass
        elif line[0].isdigit():
            size, file_name = line.split(" ")
            files.append(("".join(cwd).replace("//", "/") + file_name, int(size)))
    dir_sizes = []
    for d in directories:
        t = 0
        for f, s in files:
            if f.startswith(d):
                t += s
        dir_sizes.append((d, t))
    return sum([s for d, s in dir_sizes if s <= 100000])


example_answer = a(example)
print(example_answer)
assert example_answer == 95437
answer = a(puzzle.input_data)
print("a:", answer)
puzzle.answer_a = answer


# Part b
def b(data):
    directories = []
    files = []
    cwd = []
    lines = iter(data.splitlines())
    for line in lines:
        if line.startswith("$"):
            _, command, *arg = line.split(" ")
            if command == "cd":
                if arg[0] == "..":
                    cwd.pop()
                else:
                    directories.append(("".join(cwd) + arg[0] + "/").replace("//", "/"))
                    cwd.append(arg[0] + "/")
            elif command == "ls":
                pass
        elif line.startswith("dir "):
            pass
        elif line[0].isdigit():
            size, file_name = line.split(" ")
            files.append(("".join(cwd).replace("//", "/") + file_name, int(size)))
    dir_sizes = []
    for d in directories:
        t = 0
        for f, s in files:
            if f.startswith(d):
                t += s
        dir_sizes.append((d, t))
    total_size_bad_root = sum([s for d, s in dir_sizes])
    total_size_not_root = sum([s for d, s in dir_sizes if d != "/"])
    total_size = total_size_bad_root - total_size_not_root
    need_to_free = 70000000 - total_size
    return min([s for d, s in dir_sizes if s >= 30000000 - need_to_free])


example_answer = b(example)
print(example_answer)
assert example_answer == 24933642
answer = b(puzzle.input_data)
print("b:", answer)
puzzle.answer_b = answer
