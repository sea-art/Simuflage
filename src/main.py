from application import Application
from designpoint import Designpoint
from component import Component
import simulator

if __name__ == "__main__":
    components = [Component(80), Component(80), Component(80)]
    applications = [Application(30), Application(30), Application(30)]
    app_map = \
        [
            (components[0], applications[0]),
            (components[1], applications[1]),
            (components[2], applications[2]),

        ]

    dp = Designpoint(components, applications, app_map)

    sim = simulator.Simulator(dp)
    sim.run(500)
