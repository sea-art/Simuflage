from design.application import Application
from design.designpoint import Designpoint
from design.component import Component
from simulation import simulator
import numpy as np
import math
import random


def all_possible_pos_mappings(n):
    """ Cartesian product of all possible position values.

    :param n: amount of components
    :return: (N x 2) integer array containing all possible positions.
    """
    grid_size = int(math.ceil(math.sqrt(n)))
    x = np.arange(grid_size)

    return np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])


def random_experiment():
    """ Random experiment for the simulator.
    n components will randomly be placed on a grid with a random power capacity and a random application mapped to it.

    :return: None
    """
    n = 9
    choices = list(map(tuple, all_possible_pos_mappings(n)))

    components = []

    for x in np.random.randint(80, 120, n):
        loc = random.choice(choices)
        components.append(Component(x, loc))
        choices.remove(loc)

    applications = [Application(x) for x in np.random.randint(30, 50, n)]
    app_map = [(components[x], applications[x]) for x in range(n)]

    dp = Designpoint(components, applications, app_map)
    sim = simulator.Simulator(dp)

    sim.run(until_failure=True, debug=False)
    print("TTF", sim.iterations)


def monte_carlo(iterations=100):
    """ Run a Monte Carlo simulation to receive a MTTF of a design point.

    :param iterations: integer - number of iterations
    :return: None
    """
    TTFs = []

    dp = manual_designpoint()

    for i in range(iterations):
        sim = manual_experiment(dp)
        TTFs.append(sim.iterations)

    print("MTTF:", sum(TTFs) / len(TTFs))


def manual_designpoint():
    """ Manually create a design point.

    Function mainly used for testing purposes

    :return: designpoint object.
    """
    c1 = Component(200, (0, 1))
    c2 = Component(220, (1, 0))

    a1 = Application(50)
    a2 = Application(50)

    components = [c1, c2]
    applications = [a1, a2]
    app_map = [(c1, a1), (c2, a2)]

    dp = Designpoint(components, applications, app_map)

    return dp


def manual_experiment(dp=None, debug=False):
    """ Manual experiment for the simulator.
    Components are manually made. This function is mainly used for test/debugging purposes.

    :return: None
    """
    if not dp:
        dp = manual_designpoint()

    sim = simulator.Simulator(dp)
    sim.run(until_failure=True, debug=False)

    if debug:
        print("TTF", sim.iterations)

    return sim


if __name__ == "__main__":
    random_experiment()
