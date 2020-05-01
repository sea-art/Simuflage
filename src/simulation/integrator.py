from abc import ABC, abstractmethod
import os
import numpy as np

from simulation.elements.thermals import Thermals
from simulation.elements.agings import Agings
from simulation.elements.components import Components


class AbsIntegrator(ABC):
    @abstractmethod
    def step(self, *args):
        pass

    def print_timestep(self, timestep):
        pass

    def log_timestep(self, timestep):
        pass


class Integrator(AbsIntegrator):
    """ This class should be edited when adding new elements or changing simulation functionality.

    """
    def __init__(self, design_point):
        dp_data = design_point.to_numpy()

        # Simulation variables
        self._components = Components(dp_data[0], dp_data[2], dp_data[3], dp_data[4])
        self._thermals = Thermals(dp_data[1])
        self._agings = Agings(self._components.alive_components)

    def step(self):
        self._thermals.step(self._components.comp_loc_map)
        remap_required = self._agings.step(self._components.alive_components, self._thermals.temps)
        system_ok = self._components.step(self._agings.cur_agings)

        return system_ok

    def print_status(self, timestep):
        print(timestep)

    def log_timestep(self, timestep):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        f = open(root_dir + "/../../out/" + filename_out, "a+")
        f.write("%d %s %f\n" % (timestep, np.sum(self._components.capacities), np.average(self._thermals.temps)))
        f.close()