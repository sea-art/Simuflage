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

    @abstractmethod
    def print_status(self, timestep):
        pass

    @abstractmethod
    def log_timestep(self, timestep):
        pass


class Integrator(AbsIntegrator):
    """ This class should be edited when adding new elements or changing simulation functionality."""
    def __init__(self, design_point):
        """ Integrator is used to change/add functionality to the simulator.

        All elements of the simulator are required to work together, this class achieves this collaboration between
        simulator classes. More information is given at the CONTRIBUTING.md file.

        :param design_point: Designpoint object representing a system to evaluate.
        """
        dp_data = design_point.to_numpy()

        # Simulation variables
        self._components = Components(dp_data[0], dp_data[2], dp_data[3], dp_data[4])
        self._thermals = Thermals(dp_data[1])
        self._agings = Agings(self._components.alive_components)

    def step(self):
        """ Evaluate the simulator by one timestep.

        This function contains all integration aspects regarding the functionality per timestep of the simulator.
        Since these applications are ought to use variables from eachother, this collaboration is implemented here.
        For more information, see CONTRIBUTING.md.

        :return: Boolean indicating if a core has failed this iteration.
        """
        self._thermals.step(self._components.comp_loc_map)
        remap_required = self._agings.step(self._components.alive_components, self._thermals.temps)
        system_ok = self._components.step(self._agings.cur_agings)

        return system_ok

    def print_status(self, timestep):
        """ Print the current system values based on each simulation iteration.

        :param timestep - The current amount of timesteps that the simulator has processed.
        :return: None
        """
        print(timestep)

    def log_timestep(self, timestep):
        """ Write information about the current timestep to a file.

        :param filename_out: file to write to (/out/<filename_out>)
        :return: None
        """
        root_dir = os.path.dirname(os.path.abspath(__file__))
        f = open(root_dir + "/../../out/" + filename_out, "a+")
        f.write("%d %s %f\n" % (timestep, np.sum(self._components.capacities), np.average(self._thermals.temps)))
        f.close()
