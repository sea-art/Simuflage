from application import Application
from designpoint import Designpoint
from component import Component
import simulator

if __name__ == "__main__":
    components = [Component(100), Component(50)]
    applications = [Application(60), Application(40)]
    app_map = {components[0]: applications[0], components[1]: applications[1]}

    dp = Designpoint(components, applications, app_map)
    sim = simulator.Simulator(dp)
    sim.run(1000)
