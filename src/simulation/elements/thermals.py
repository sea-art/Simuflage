#!/usr/bin/env python

""" Contains all logical operations that are required for the thermals of components in the simulation

Thermals are stored as a 2D numpy float array.
"""

import numpy as np
from scipy import signal

from simulation.elements.element import SimulatorElement

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"

ENV_TEMP = 20


class Thermals(SimulatorElement):
    """ Contains all logical operators based on the thermals of components."""

    def __init__(self, workload, max_temps, comp_loc_map):
        """ Initializes a Thermals object based on the current initial thermals.

        :param workload: 2D numpy float array with values between [0., 1.], indicating workload.
        :param max_temps: 2D numpy float array locally indicating the max temp (at 1.0 usage) per component.
        :param comp_loc_map: mapping of component index to xy-location (i, x , y)
        """
        self._max_temps = max_temps
        self._temps = ENV_TEMP + max_temps * workload
        self._m = comp_loc_map  # comp_loc_map

    @property
    def temps(self):
        """ Getter function for the temps instance variable.

        :return: 2D float numpy array with temperatures on component positions.
        """
        return self._temps

    def adjusted_thermals(self, workload, fluc):
        """ Adjusts the thermals based on uniform fluctuation and neighbour thermal influences.

        :param workload: 2D numpy float array with values between [0., 1.], indicating workload.
        :param fluc: float representing the max uniformly fluctuation of temperature.
        :return: None
        """
        temperatures = workload * self._max_temps
        temperatures[self._m['y'], self._m['x']] += \
            np.random.uniform(-fluc, fluc, temperatures.shape)[self._m['y'], self._m['x']]

        neighbour_thermals = self.neighbour_thermal_influences(temperatures)

        self._temps[self._m['y'], self._m['x']] = neighbour_thermals[self._m['y'], self._m['x']]

    @staticmethod
    def neighbour_thermal_influences(temperatures, kernel=None):
        """ Adjusts the thermals based on the neighbouring components thermals

        :param temperatures: 2D numpy float array - locally indicating the temperatures
        :param kernel: 2D kernel which will be used for convolution
        :return: 2D numpy float array - grid thermals after neighbouring contributions
        """
        if not kernel:
            kernel = np.asarray([[0.01, 0.01, 0.01],
                                 [0.01, 1,    0.01],
                                 [0.01, 0.01, 0.01]])

        return signal.convolve2d(temperatures, kernel, "same")

    def step(self, workload, fluctuate=0.0):
        """ Iterate the thermal influences

        :param workload: 2D numpy float array with values between [0., 1.], indicating workload.
        :param fluctuate: (float) representing the max uniformly fluctuation of temperature each iteration.
        :return: None
        """
        self.adjusted_thermals(workload, fluctuate)

    def do_n_steps(self, n, workload, fluctuate=0.0):
        """ Run n iterations regarding the thermals of the simulation.

        :param n: (int) amount of timesteps
        :param workload: 2D numpy float array with values between [0., 1.], indicating workload.
        :param fluctuate: (float) representing the max uniformly fluctuation of temperature each iteration.
        :return: None
        """
        self.step(workload, fluctuate=fluctuate)
