from application import Application
from designpoint import Designpoint
from component import Component
import simulator
import numpy as np

if __name__ == "__main__":
    n = np.random.randint(1, 100)

    components = [Component(x) for x in np.random.randint(30, 60, n)]
    applications = [Application(x) for x in np.random.randint(1, 29, n)]
    app_map = [(components[x], applications[x]) for x in range(n)]

    dp = Designpoint(components, applications, app_map)

    sim = simulator.Simulator(dp)
    sim.run(5000)
