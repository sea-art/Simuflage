from design.application import Application
from design.designpoint import Designpoint
from design.component import Component
from simulation import simulator
import numpy as np


def random_experiment():
    n = 5

    components = [Component(x, (0, 0)) for x in np.random.randint(50, 60, n)] # TODO: random loc without duplicates
    applications = [Application(x) for x in np.random.randint(5, 30, n)]
    app_map = [(components[x], applications[x]) for x in range(n)]

    dp = Designpoint(components, applications, app_map)

    sim = simulator.Simulator(dp)

    sim.run(until_failure=True)


def manual_experiment():
    c1 = Component(100, (0, 1))
    c2 = Component(100, (1, 0))

    a1 = Application(50)
    a2 = Application(50)

    components = [c1, c2]
    applications = [a1, a2]
    app_map = [(c1, a2), (c2, a2)]

    dp = Designpoint(components, applications, app_map)

    sim = simulator.Simulator(dp)

    # print()

    # sim.run(until_failure=True, debug=True)


if __name__ == "__main__":
    manual_experiment()
