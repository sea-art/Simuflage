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


def manual_designpoint(cap1=100, cap2=100,
              loc1=(0,0), loc2=(1,1),
              app1=50, app2=50,
              policy='random'):
    c1 = Component(cap1, loc1)
    c2 = Component(cap2, loc2)

    a1 = Application(app1)
    a2 = Application(app2)

    mapping = [(c1, a1), (c2, a2)]

    return Designpoint([c1, c2], [a1, a2], mapping, policy=policy)


def monte_carlo(designpoints, iterations=1000):
    TTFs = [[] for _ in designpoints]

    for a in range(iterations):
        i = random.randint(0, len(designpoints) - 1)
        TTFs[i].append(Simulator(designpoints[i]).run_optimized())

    for i in range(len(TTFs)):
        TTFs[i] = sum(TTFs[i]) / len(TTFs[i])

    print(TTFs)


def run_test():
    # dp1 = create_dp(cap1=100, cap2=100, policy='most')
    # dp2 = create_dp(cap1=100, cap2=100, policy='least')

    dp1 = random_designpoint()
    dp2 = random_designpoint()

    dp3 = random_designpoint()
    dp4 = random_designpoint()

    print("dp1", dp1)
    print("dp2", dp2)
    print("dp3", dp3)
    print("dp4", dp4)

    monte_carlo([dp1, dp2, dp3, dp4])


if __name__ == "__main__":
    run_test()