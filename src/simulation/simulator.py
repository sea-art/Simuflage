import sys

from simulation.integrator import Integrator


class Simulator:
    def __init__(self, design_point):
        """Creates a simulator to calculate the TTF, temperatures and power output given a designpoint.

        :param design_point: Designpoint object representing a system to evaluate.
        """
        dp_data = design_point.to_numpy()

        self._integrator = Integrator(design_point)
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
        self._integrator.log_timestep(self._timesteps)

    def print_current_status(self):
        """ Print the current system values based on each simulation iteration.

        :return: None
        """
        self._integrator.print_status(self._timesteps)

    def step(self):
        """ Run one iteration of the simulator.

        :return: boolean indicating if a core has failed this iteration.
        """
        system_ok = self._integrator.step()

        self._timesteps += 1

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
