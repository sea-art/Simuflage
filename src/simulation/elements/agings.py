#!/usr/bin/env python

""" Contains all logical operations that are required for the aging of components in the simulation

Agings are stored as a 2D numpy float array.
"""

import numpy as np

from .element import SimulatorElement
from simulation.faultmodels import electro_migration

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


class Agings(SimulatorElement):
    """ Contains all logical operators based on the aging of components."""

    def __init__(self, alive_components, temperatures, workload, model=electro_migration):
        """ Initializes an aging grid (which computes the aging rate for each of the components) based on the Weibull
        distribution.

        :param alive_components: 2D numpy boolean array (True indicates a living component on that position)
        """
        samples = np.zeros(alive_components.shape)
        samples[alive_components] = model(temperatures[alive_components]) * np.random.weibull(5.0,
                                                                                              np.sum(alive_components))

        self._lambdas = np.divide(1, np.floor(samples), out=np.zeros_like(samples), where=samples != 0)
        self._wear = np.zeros(alive_components.shape, dtype=np.float)

        self._cur_workload = np.copy(workload)  # Stores the current workload to see alterations in remapping
        self._model = model

    def __repr__(self):
        """ Representation of an Agings object.

        :return: string - representation of this Agings object
        """
        return "Agings: " + str(self._wear)

    @property
    def cur_agings(self):
        """ Getter function for the cur_agings instance variable.

        :return: 2D float numpy array with the current agings based on component positions.
        """
        return self._wear

    def resample_workload_changes(self, workload, thermals):
        """ Resample the next failure of components based on the given workload and thermals.

        :param workload: 2D numpy float array with values between [0.0, 1.0], indicating workload.
        :param thermals: 2D numpy float array with the current local thermals at this iteration.
        :return:
        """
        remapped_locs = workload > self._cur_workload
        samples = self._model(thermals[remapped_locs]) * np.random.weibull(5, np.sum(remapped_locs))

        self._lambdas[remapped_locs] = np.divide(1, samples)
        self._cur_workload = np.copy(workload)

    def _update_agings(self, alive_components, thermals):
        """ Update the aging values of all components with a single iteration.

        :return: Boolean indicating if any new failures have occurred (which should be handled).
        """
        assert np.all(self._lambdas[alive_components] >= 0), "Negative aging rate"

        self._wear[alive_components] += self._lambdas[alive_components]

        return np.any(self._wear[alive_components] > 0.9999)

    def steps_till_next_failure(self, alive_components, thermals, steps_taken):
        """ Calculate in how many timesteps the next failure occurs.

        :param alive_components: 2D numpy boolean array indicating the position of alive components.
        :param thermals: 2D numpy float array with the current local thermals at this iteration.
        :param steps_taken: integer - indicating how many timesteps are already taken in the simulation.
        :return: int - indicating how many timesteps are required for the next failure.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            timesteps = np.ceil((1.0 - self._wear) / self._lambdas)
            timesteps[np.isinf(timesteps)] = 0

        return int(np.amin(timesteps[alive_components]))

    def step(self, alive_components, thermals):
        """ Increment a single timestep regarding the aging process of the simulation

        :param alive_components: 2D numpy boolean array indicating the position of alive components.
        :param thermals: 2D numpy float array with the current local thermals at this iteration.
        :return: Boolean - indicating if any new failures have occurred (which should be handled).
        """
        return self._update_agings(alive_components, thermals)

    def step_till_failure(self, n, alive_components, thermals):
        """ Increment n timesteps regarding te aging process of the simulation.

        :param n: amount of timesteps to take
        :param alive_components: 2D numpy boolean array indicating the position of alive components.
        :param thermals: 2D numpy float array with the current local thermals at this iteration.
        :return: Boolean - indicating if any new failures have occurred (which should be handled).
        """
        assert n > 0, "Incrementing with 0 timesteps\n" + str(self._wear)

        self._wear[alive_components] += n * self._lambdas[alive_components]

        assert np.any(self._wear[alive_components] > 0.9999), \
            "n steps did not result in aging > 1.0\n" + str(self._wear[alive_components])

        return np.any(self._wear[alive_components] > 0.9999)

    def reset(self, alive_components, temperatures, workload):
        """ Resets the agings back to default.

        :return: None
        """
        samples = np.zeros(alive_components.shape)
        samples[alive_components] = self._model(temperatures[alive_components]) * \
            np.random.weibull(5.0, np.sum(alive_components))

        self._lambdas = np.divide(1, np.floor(samples), out=np.zeros_like(samples), where=samples != 0)
        self._wear = np.zeros(alive_components.shape, dtype=np.float)
        self._cur_workload = np.copy(workload)
