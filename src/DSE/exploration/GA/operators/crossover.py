import numpy as np


def one_point_crossover(parent1, parent2):
    max_idx = min(len(parent1), len(parent2))
    try:
        point = np.random.randint(1, max_idx)
    except ValueError:  # error thrown when max_idx = 1, solved by setting manually.
        point = 1

    return np.append(parent1[:point], parent2[point:]), \
        np.append(parent2[:point], parent1[point:])


def two_point_crossover(parent1, parent2):
    max_idx = min(len(parent1), len(parent2))

    point1 = np.random.randint(1, max_idx)
    point2 = np.random.randint(1, max_idx - 1)

    if point2 >= point1:
        point2 += 1
    else:
        point1, point2 = point2, point1

    return np.append(parent1[:point1], parent2[point1:point2], parent1[point2:]), \
        np.append(parent2[:point1], parent1[point1:point2], parent2[point2:])
