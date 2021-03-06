#!/usr/bin/env python

""" Contains all logical operations that are required for the thermals of components in the simulation

Thermals are stored as a 2D numpy float array.
"""

import numpy as np
from scipy import signal

from .element import SimulatorElement

ENV_TEMP = 15
IDLE_TEMP = 5


class Thermals(SimulatorElement):
    """ Contains all logical operators based on the thermals of components."""

    def __init__(self, workload, max_temps, comp_loc_map, alive_components):
        """ Initializes a Thermals object based on the current initial thermals.

        :param workload: 2D numpy float array with values between [0., 1.], indicating workload.
        :param max_temps: 2D numpy float array locally indicating the max temp (at 1.0 usage) per component.
        :param comp_loc_map: mapping of component index to xy-location (i, x , y)
        """
        self._max_temps = max_temps
        self._m = comp_loc_map
        self._alive_components = alive_components

        self._temps = np.zeros(workload.shape)
        self._temps[alive_components] = self._adjusted_thermals(workload, 0.0)[alive_components]

    def __repr__(self):
        """ Representation of an Components object.

        :return: string - representation of this Components object
        """
        return str(self._temps)

    @property
    def temps(self):
        """ Getter function for the temps instance variable.

        :return: 2D float numpy array with temperatures on component positions.
        """
        return self._temps

    def _adjusted_thermals(self, workload, fluc):
        """ Adjusts the thermals based on uniform fluctuation and neighbour thermal influences.

        :param workload: 2D numpy float array with values between [0., 1.], indicating workload.
        :param fluc: float representing the max uniformly fluctuation of temperature.
        :return: None
        """
        # Assign temperatures to components based on workload, max_temps and environmental temperature.
        temperatures = ENV_TEMP + workload * self._max_temps

        # Assigns the idle temperature to alive_components that have no workload.
        temperatures[self._alive_components] = np.where(workload == 0., IDLE_TEMP, temperatures)[self._alive_components]

        # Increases the heat of components based on other adjacent component thermals.
        neighbour_thermals = self._neighbour_thermal_influences(temperatures)

        return neighbour_thermals

    @staticmethod
    def _neighbour_thermal_influences(temperatures, kernel=None):
        """ Adjusts the thermals based on the neighbouring components thermals

        :param temperatures: 2D numpy float array - locally indicating the temperatures
        :param kernel: 2D kernel which will be used for convolution
        :return: 2D numpy float array - grid thermals after neighbouring contributions
        """
        ni = 0.1

        if not kernel:
            kernel = np.asarray([[ni, ni, ni],
                                 [ni, 1, ni],
                                 [ni, ni, ni]])

        return signal.convolve2d(temperatures, kernel, "same")

    def step(self, workload, fluctuate=0.0):
        """ Iterate the thermal influences

        :param workload: 2D numpy float array with values between [0., 1.], indicating workload.
        :param fluctuate: (float) representing the max uniformly fluctuation of temperature each iteration.
        :return: None
        """
        self._temps = np.zeros(workload.shape)
        self._temps[self._alive_components] = self._adjusted_thermals(workload, fluctuate)[self._alive_components]

    def step_till_failure(self, n, workload, fluctuate=0.0):
        """ Run n sample_budget regarding the thermals of the simulation.

        :param n: (int) amount of timesteps
        :param workload: 2D numpy float array with values between [0., 1.], indicating workload.
        :param fluctuate: (float) representing the max uniformly fluctuation of temperature each iteration.
        :return: None
        """

        self.step(workload, fluctuate=fluctuate)

    def reset(self, workload, alive_components):
        """ Resets the thermals back to default.

        :return: None
        """
        self._temps = np.zeros(workload.shape)
        self._alive_components = alive_components
        self._temps[alive_components] = self._adjusted_thermals(workload, 0.0)[alive_components]
