import numpy as np
from aocd.models import Puzzle

YEAR = 2019
DAY = 8

puzzle = Puzzle(year=YEAR, day=DAY)


# Part a
def a(data):
    image = np.array(list(map(int, data)))
    image.shape = (-1, 6, 25)
    zeros = np.sum(image == 0, axis=(1, 2))
    layer_idx = zeros.argmin()
    return (image[layer_idx, ...] == 1).sum() * (image[layer_idx, ...] == 2).sum()

answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 2080

# Part b
def b(data):
    image = np.array(list(map(int, data)))
    image.shape = (-1, 6, 25)
    import matplotlib.pyplot as plt
    out = np.zeros((6, 25))
    layer_idx = (image < 2).argmax(axis=0)
    for i in range(6):
        for j in range(25):
            out[i, j] = image[layer_idx[i, j], i, j]
    plt.imshow(out)
    plt.show()


b(puzzle.input_data)
