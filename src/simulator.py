import numpy as np
import os
import sys

from designpoint import Designpoint
from thermals import Thermals
from agings import Agings


class Simulator:
    def __init__(self, design_point):
        """Creates a simulator to calculate the TTF, temperatures and power output given a designpoint.

        :param design_point: Designpoint object representing a system to evaluate.
        """
        dp_data = design_point.to_numpy()

        # Simulation variables
        self.capacities = dp_data[0]
        self.thermals = Thermals(dp_data[1])
        self.power_uses = dp_data[2]
        self.alive_components = self.capacities > 0
        self.agings = Agings(self.alive_components)

        # Mappings
        self.comp_loc_map = dp_data[3]
        self.app_mapping = dp_data[4]

        # Miscellaneous variables
        self.nr_applications = np.count_nonzero(self.app_mapping)
        self.nr_components = np.count_nonzero(self.capacities)
        self.iterations = 0

    def adjust_power_uses(self):
        """Updates the power_uses for components based on the application mapping (self.app_mapping)."""
        foo = np.zeros(self.nr_components)

        for a, b in self.app_mapping:
            foo[a] += b

        self.power_uses = foo

    def clear_failed_components(self, new_failed_components):
        """ Cleanup and alter variables of new components that have failed.

        :param new_failed_components: Numpy boolean array indicating which components have failed.
        :return: Numpy array of indices of failed components
        """
        # TODO: remove from comp to loc mapping?
        self.failed_components[new_failed_components] = True
        self.capacities[new_failed_components] = 0

        alive_indices = np.nonzero(np.invert(new_failed_components))[0]

        if alive_indices.size > 0:
            self.comp_loc_map = self.comp_loc_map[alive_indices]

        return np.nonzero(new_failed_components)

    def handle_failures(self):
        """Look at the aging values to determine if a component has failed.
        If a component has failed, the applications that were mapped to that component will randomly be remapped
        to components with a sufficient amount of slack.

        :return: Boolean indicating failure occurrence.
        """
        all_failed_components = self.agings >= 1.0  # All failed components

        # Check if failed components are already adjusted (i.e. any remapping required?)
        if np.any(all_failed_components[np.invert(self.failed_components)]):
            failed_indices = self.clear_failed_components(all_failed_components)

            # All applications that have to be remapped
            to_map = self.app_mapping[np.isin(self.app_mapping['comp'], failed_indices[0])]

            # Removes all applications that are mapped towards failed components
            self.app_mapping = self.app_mapping[np.isin(self.app_mapping['comp'], failed_indices[0], invert=True)]
            self.adjust_power_uses()

            for app in to_map['app']:
                components_slack = self.capacities - self.power_uses

                # Loop randomly over all non-failed components
                for i in np.random.permutation(np.arange(self.nr_components)[np.invert(self.failed_components)]):

                    if app <= components_slack[i]:
                        self.app_mapping = np.append(self.app_mapping, np.array([(i, app)],
                                                                                dtype=self.app_mapping.dtype))
                        self.adjust_power_uses()  # TODO: can be speed up
                        break

            self.print_current_status()

        if self.app_mapping.size != self.nr_applications:
            print("required", app)
            print("available", self.capacities - self.power_uses)

        return self.app_mapping.size == self.nr_applications  # checks if all applications are mapped, otherwise failure

    def log_iteration(self, filename_out):
        """ Write the current iteration information to a file.

        :param filename_out: file to write to (/out/<filename_out>)
        :return: None
        """
        root_dir = os.path.dirname(os.path.abspath(__file__))
        f = open(root_dir + "/../out/" + filename_out, "a+")
        f.write("%d %d %f\n" % (self.iterations, np.sum(self.capacities), np.average(self.isolated_thermals)))
        f.close()

    def print_current_status(self):
        print("i:", self.iterations, "-", np.nonzero(self.failed_components)[0].size, "core(s) have failed")

        with np.errstate(divide='ignore', invalid='ignore'):
            grid = self.power_uses / self.capacities

        print("grid:\n", grid)
        print("Thermals:\n", self.thermals)
        print("Application mapping:\n", np.sort(self.app_mapping), "\n")

    def iterate(self):
        """ Run one iteration within the simulator.

        :return: boolean indicating if a core has failed this iteration.
        """
        self.update_agings()
        self.adjusted_thermals(0.1)

        self.log_iteration("a.txt")
        self.iterations += 1
        return self.handle_failures()

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
