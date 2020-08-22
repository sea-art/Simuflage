#!/usr/bin/env python

""" This file contains most top-level functionality in order to evaluate a designpoint.

Functionalities of the simulator should be defined in the integrator.py file.
"""

import sys

from .integrator import Integrator, AbsSimulator


__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


class Simulator(AbsSimulator):
    """ This file contains all the functions that the Simulator is ought to perform."""

    def __init__(self, design_point):
        """Creates a simulator to calculate the TTF, temperatures and power output given a designpoint.

        :param design_point: DesignPoint object representing a system to evaluate.
        """
        self._integrator = Integrator(design_point)
        self._timesteps = 0

    @property
    def timesteps(self):
        """ Getter function for the cur_agings instance variable.

        :return: 2D float numpy array with the current agings based on component positions.
        """
        return self._timesteps

    def log_timestep(self, filename_out):
        """ Write the current iteration information to a file.

        :param filename_out: file to write to (/out/<filename_out>)
        :return: None
        """
        self._integrator.log_timestep(filename_out)

    def print_status(self):
        """ Print the current system values based on each simulation iteration.

        :return: None
        """
        self._integrator.print_status()

    def step(self):
        """ Run one iteration of the simulator.

        :return: Boolean - indicating if the system is still running.
        """
        system_ok = self._integrator.step()

        self._timesteps += 1

        return system_ok

    def step_till_failure(self):
        """ Run n iterations of the simulator.

        :return: Boolean - indicating if the system is still running.
        """
        system_ok = self._integrator.step_till_failure()
        self._timesteps = self._integrator.timesteps

        return system_ok

    def reset(self):
        """ Resets the simulator back to default.

        :return: None
        """
        self._integrator.reset()
        self._timesteps = 0

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
                self.print_status()

            if not self.step():
                return self._timesteps

    def run_optimized(self):
        """ Run the simulation in an optimized form, where timesteps will be skipped to
        the point where a failure occurs.

        :return: integer - timesteps till failure of the design point
        """
        while True:
            if not self.step_till_failure():
                break

        avg_watt_used = self._integrator.total_watt_used / self._timesteps + self._integrator.grid_size

        ts = self._timesteps
        self.reset()

        return ts, avg_watt_used, self._integrator.grid_size
