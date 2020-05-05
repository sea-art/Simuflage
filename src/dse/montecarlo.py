import random
import math
from copy import deepcopy

import numpy as np

from design.application import Application
from design.component import Component
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

def random_designpoint():
    """ Random experiment for the simulator.
    n components will randomly be placed on a grid with a random power capacity and a random application mapped to it.

    :return: None
    """
    n = random.randint(2, 20)
    choices = list(map(tuple, all_possible_pos_mappings(n)))

    components = []

    for x in np.random.randint(61, 200, n):
        loc = random.choice(choices)
        components.append(Component(x, loc))
        choices.remove(loc)

    applications = [Application(x) for x in np.random.randint(10, 60, n)]
    app_map = [(components[x], applications[x]) for x in range(n)]

    policy = random.choice(["random", "most", "least"])

    return Designpoint(components, applications, app_map, policy)


def manual_designpoint(caps, locs, apps, maps, policy='random'):
    comps = [Component(caps[i], locs[i]) for i in range(len(caps))]
    apps = [Application(a) for a in apps]
    mapping = [(comps[maps[i][0]], apps[maps[i][1]]) for i in range(len(maps))]

    return Designpoint(comps, apps, mapping, policy=policy)


def monte_carlo(designpoints, iterations=1000):
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

    dp1 = random_designpoint()
    dp2 = random_designpoint()

    # dp1 = manual_designpoint(caps=[100, 100, 100, 100],
    #                          locs=[(0, 0), (2, 0), (0, 2), (2, 2)],
    #                          apps=[20, 70, 10, 80],
    #                          maps=[(0, 0), (0, 1), (1, 2), (1, 3)],
    #                          policy='random')
    #
    # dp2 = manual_designpoint(caps=[100, 100, 100, 100],
    #                          locs=[(0, 0), (2, 0), (0, 2), (2, 2)],
    #                          apps=[20, 70, 10, 80],
    #                          maps=[(0, 0), (1, 1), (2, 2), (3, 3)],
    #                          policy='random')

    dp3 = random_designpoint()
    dp4 = random_designpoint()

    x = [random_designpoint() for _ in range(20)]

    TTFs = monte_carlo(x)

    print(sorted(TTFs))



if __name__ == "__main__":
    run_test()