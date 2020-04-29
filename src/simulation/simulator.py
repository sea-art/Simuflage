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
        self.components = Components(dp_data[0], dp_data[2], dp_data[3], dp_data[4])
        self.thermals = Thermals(dp_data[1])
        self.agings = Agings(self.components.alive_components)

        self.iterations = 0

    def log_iteration(self, filename_out):
        """ Write the current iteration information to a file.

        :param filename_out: file to write to (/out/<filename_out>)
        :return: None
        """
        root_dir = os.path.dirname(os.path.abspath(__file__))
        f = open(root_dir + "/../../out/" + filename_out, "a+")
        f.write("%d %d %f\n" % (self.iterations, np.sum(self.capacities), np.average(self.isolated_thermals)))
        f.close()

    def print_current_status(self):
        """ Print the current system values based on each simulation iteration.

        :return: None
        """
        # print("i:", self.iterations, "-", np.count_nonzero(self.components.alive_components), "core(s) have failed")
        # print("iteration: ", self.iterations)
        print((self.components.power_uses / self.components.capacities) * 100)

        # print(self.components.alive_components)

        # with np.errstate(divide='ignore', invalid='ignore'):
        #     grid = self.components.power_uses / self.components.capacities
        #
        # print("grid:\n", grid)
        # print("Thermals:\n", self.thermals.temps)
        # print("Application mapping:\n", np.sort(self.components.app_mapping), "\n")

    def iterate(self):
        """ Run one iteration of the simulator.

        :return: boolean indicating if a core has failed this iteration.
        """

        keep_iterating = True

        self.thermals.iterate(self.components.comp_loc_map)
        self.agings.iterate(self.components.alive_components, self.thermals.temps)
        keep_iterating = self.components.iterate(self.agings.cur_agings)

        # self.log_iteration("a.txt")
        self.iterations += 1

        return keep_iterating

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

            if not self.iterate():
                print("failed - TTF:", self.iterations)
                return self.iterations
