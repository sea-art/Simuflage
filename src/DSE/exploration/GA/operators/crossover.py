import numpy as np
import random

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


def swap_elements(child, indices, possible_swaps):
    """ Swap the elements on the indices of the child with the given
    possible values.

    :param child: container where the replaced items will be swapped
    :param indices: indices of the items in child to swap
    :param possible_swaps: collection of possible values to swap
    :return: child - with items on incides randomly swapped with possible values.
    """
    for i in indices:
        new_loc1 = random.choice(list(possible_swaps))
        child[i] = new_loc1
        possible_swaps.remove(new_loc1)

    return child


def one_point_swapover(parent1, parent2):
    """ Swap k random elements with unique elements from the other parent.

    Given two parents, will swap k random elements between each other while
    remaining uniqueness in the genes of the chromosome.

    :param parent1: iterable individual.
    :param parent2: iterable individual.
    :return: child1, child2 with k genes swapped while remaining unique elements in chromosome.
    """
    # Find all unique locations from the other set that are not in current set.
    s1, s2 = set(parent1), set(parent2)
    possible_locs1, possible_locs2 = s2 - s1, s1 - s2

    # Determine randomly the amount of locations to swap.
    k1, k2 = 0, 0
    if len(possible_locs1) > 0:
        k1 = random.randint(1, len(possible_locs1) - 1)
    if len(possible_locs2) > 0:
        k2 = random.randint(1, len(possible_locs2) - 1)

    print("Swapping", k1, k2)

    # Indices that will be swapped
    indices1 = np.random.choice(range(len(s1)), k1, replace=False)
    indices2 = np.random.choice(range(len(s2)), k2, replace=False)
    child1, child2 = parent1[:], parent2[:]

    return swap_elements(child1, indices1, possible_locs1), \
        swap_elements(child2, indices2, possible_locs2)

