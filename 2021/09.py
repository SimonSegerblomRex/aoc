import numpy as np
from aocd.models import Puzzle
from scipy.ndimage import label, minimum_filter

YEAR = 2021
DAY = 9

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    height_map = np.array(
        [[int(d) for d in line] for line in data.splitlines()], dtype=int
    )
    min_value = minimum_filter(height_map, size=(3, 3))
    height_map_plus_1 = height_map + 1
    min_value_plus_1 = minimum_filter(height_map_plus_1, size=(3, 3))
    low_points = (height_map == min_value) & (height_map_plus_1 == min_value_plus_1)
    return height_map_plus_1[low_points].sum()


example_answer = a(puzzle.example_data)
print(example_answer)
assert example_answer == 15
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 478


# Part b
def b(data):
    height_map = np.array(
        [[int(d) for d in line] for line in data.splitlines()], dtype=int
    )
    image = np.ones(height_map.shape, dtype=np.uint8)
    image[height_map == 9] = 0
    labeled, _ = label(image)
    sizes = np.bincount(labeled.flatten())[1:]
    return np.prod(sorted(sizes)[-3:])


example_answer = b(puzzle.example_data)
print(example_answer)
assert example_answer == 1134
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 1327014
