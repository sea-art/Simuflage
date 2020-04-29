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

    for x in np.random.randint(100, 150, n):
        loc = random.choice(choices)
        components.append(Component(x, loc))
        choices.remove(loc)

    applications = [Application(x) for x in np.random.randint(1, 100, n)]
    app_map = [(components[x], applications[x]) for x in range(n)]

    dp = Designpoint(components, applications, app_map)
    sim = simulator.Simulator(dp)

    sim.run(until_failure=True, debug=True)


def manual_experiment():
    """ Manual experiment for the simulator.
    Components are manually made. This function is mainly used for test/debugging purposes.

    :return: None
    """
    c1 = Component(100, (1, 1))
    c2 = Component(100, (0, 1))

    a1 = Application(50)
    a2 = Application(50)

    components = [c1, c2]
    applications = [a1, a2]
    app_map = [(c1, a1), (c2, a2)]

    dp = Designpoint(components, applications, app_map)

    sim = simulator.Simulator(dp)

    sim.run(until_failure=True, debug=True)


if __name__ == "__main__":
    random_experiment()
