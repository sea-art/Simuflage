import random
import math
import numpy as np

from design.designpoint import Designpoint
from simulation.simulator import Simulator


def all_possible_pos_mappings(n):
    """ Cartesian product of all possible position values.

    :param n: amount of components
    :return: (N x 2) integer array containing all possible positions.
    """
    grid_size = int(math.ceil(math.sqrt(n)))
    x = np.arange(grid_size)

    return np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])


def monte_carlo(designpoints, iterations=1000):
    TTFs = [[] for _ in designpoints]
    sims = [Simulator(d) for d in designpoints]

    for a in range(iterations):
        print("MC iteration:", a, end="\r")
        i = random.randint(0, len(designpoints) - 1)
        TTFs[i].append(sims[i].run_optimized())

    for i in range(len(TTFs)):
        TTFs[i] = sum(TTFs[i]) / len(TTFs[i])

    return TTFs


def run_test():
    dp1 = Designpoint.create_random(10)
    dp2 = Designpoint.create_random(10)

    MTTFs = monte_carlo([dp1, dp2])

    print("Two random designpoint MTTFs:", MTTFs)
    print("Winner is designpoint", MTTFs.index(max(MTTFs)) + 1)


if __name__ == "__main__":
    run_test()