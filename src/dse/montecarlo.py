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


def monte_carlo(designpoints, iterations=500):
    TTFs = [[] for _ in designpoints]
    sims = [Simulator(d) for d in designpoints]

    for a in range(iterations):
        i = random.randint(0, len(designpoints) - 1)
        TTFs[i].append(sims[i].run_optimized())

    for i in range(len(TTFs)):
        TTFs[i] = sum(TTFs[i]) / len(TTFs[i])

    return TTFs


def run_test():
    # dp1 = create_dp(cap1=100, cap2=100, policy='most')
    # dp2 = create_dp(cap1=100, cap2=100, policy='least')

    dp1 = Designpoint.create_random(10)
    dp2 = Designpoint.create_random(10)

    # dp1 = Designpoint.create(caps=[100, 100, 100, 100],
    #                          locs=[(0, 0), (2, 0), (0, 2), (2, 2)],
    #                          apps=[20, 70, 10, 80],
    #                          maps=[(0, 0), (0, 1), (1, 2), (1, 3)],
    #                          policy='random')
    #
    # dp2 = Designpoint.create(caps=[100, 100, 100, 100],
    #                          locs=[(0, 0), (2, 0), (0, 2), (2, 2)],
    #                          apps=[20, 70, 10, 80],
    #                          maps=[(0, 0), (1, 1), (2, 2), (3, 3)],
    #                          policy='random')

    # dp3 = random_designpoint()
    # dp4 = random_designpoint()

    TTFs = monte_carlo([dp1, dp2])

    print(sorted(TTFs))

if __name__ == "__main__":
    run_test()