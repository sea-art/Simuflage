import numpy as np
import os
import sys

from simulation.thermals import Thermals
from simulation.agings import Agings
from simulation.components import Components


class Simulator:
    def __init__(self, design_point):
        """Creates a simulator to calculate the TTF, temperatures and power output given a designpoint.

        :param design_point: Designpoint object representing a system to evaluate.
        """
        dp_data = design_point.to_numpy()

        # Simulation variables
        self._components = Components(dp_data[0], dp_data[2], dp_data[3], dp_data[4])
        self._thermals = Thermals(dp_data[1])
        self._agings = Agings(self._components.alive_components)

        self._timesteps = 0

    @property
    def timesteps(self):
        def cur_agings(self):
            """ Getter function for the cur_agings instance variable.

            :return: 2D float numpy array with the current agings based on component positions.
            """
        return self._timesteps

    def log_iteration(self, filename_out):
        """ Write the current iteration information to a file.

        :param filename_out: file to write to (/out/<filename_out>)
        :return: None
        """
        root_dir = os.path.dirname(os.path.abspath(__file__))
        f = open(root_dir + "/../../out/" + filename_out, "a+")
        f.write("%d %s %f\n" % (self._timesteps, np.sum(self._components.capacities), np.average(self._thermals.temps)))
        f.close()

    def print_current_status(self):
        """ Print the current system values based on each simulation iteration.

        :return: None
        """
        print((self._components.power_uses / self._components.capacities) * 100)

    def handle_failure_occurrence(self):
        """ This function should be called when a component has failed.

        :return:
        """
        pass

    def step(self):
        """ Run one iteration of the simulator.

        :return: boolean indicating if a core has failed this iteration.
        """
        self._timesteps += 1

        self._thermals.step(self._components.comp_loc_map)
        remap_required = self._agings.step(self._components.alive_components, self._thermals.temps)
        system_ok = self._components.step(self._agings.cur_agings)

        # self.log_iteration("a.txt")

        return system_ok

    def run(self, iteration_amount=5000, until_failure=False, debug=False):
        """ Runs the simulation an iteration_amount of time or until a failure occurs.

        :param iteration_amount:
        :param until_failure: boolean to infinitely simulate until failure
        :return: amount of iterations when the system has failed
        """
        if until_failure:
            iteration_amount = sys.maxsize

        for _ in range(iteration_amount):
            if debug:
                self.print_current_status()

            if not self.step():
                return self._timesteps
