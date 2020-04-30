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
        f.write("%d %s %f\n" % (self.iterations, np.sum(self.components.capacities), np.average(self.thermals.temps)))
        f.close()

    def print_current_status(self):
        """ Print the current system values based on each simulation iteration.

        :return: None
        """
        print((self.components.power_uses / self.components.capacities) * 100)

    def iterate(self):
        """ Run one iteration of the simulator.

        :return: boolean indicating if a core has failed this iteration.
        """
        self.iterations += 1

        keep_iterating = True

        keep_iterating &= self.thermals.iterate(self.components.comp_loc_map)
        keep_iterating &= self.agings.iterate(self.components.alive_components, self.thermals.temps)
        keep_iterating &= self.components.iterate(self.agings.cur_agings, self.iterations)

        # self.log_iteration("a.txt")

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
                return self.iterations
