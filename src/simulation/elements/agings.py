import numpy as np
from simulation.elements.element import SimulatorElement
import json


class Agings(SimulatorElement):
    """ Contains all logical operators based on the aging of components."""
    def __init__(self, alive_components):
        """ Initializes an aging grid (which computes the aging rate for each of the components) based on the Weibull
        distribution.

        :param alive_components: 2D numpy boolean array (True indicates a living component on that position)
        """
        self._omegas = np.zeros(alive_components.shape)
        # repr. iterations on 100% usage when cpu will fail
        self._omegas[alive_components] = 100 * np.random.weibull(5, np.sum(alive_components))

        self._lambdas = np.divide(1, self._omegas, out=np.zeros_like(self._omegas), where=self._omegas != 0)
        self._cur_agings = np.zeros(alive_components.shape, dtype=np.float)  # Will increment each iteration

    def __str__(self):
        return str(self._cur_agings)

    def __repr__(self):
        return self.__str__()

    @property
    def cur_agings(self):
        """ Getter function for the cur_agings instance variable.

        :return: 2D float numpy array with the current agings based on component positions.
        """
        return self._cur_agings

    def update_agings(self, alive_components, thermals):
        """ Update the aging values of all components with a single iteration.

        :return: Boolean indicating if any new failures have occurred (which should be handled).
        """

        assert np.all((self._lambdas * thermals / 100)[alive_components] >= 0), "Negative aging rate"

        self._cur_agings[alive_components] += (self._lambdas * thermals / 100)[alive_components]

        return np.any((self._cur_agings >= 1.0)[alive_components])

    def steps_till_next_failure(self, alive_components, thermals, steps_taken):
        timesteps = np.ceil(self._omegas[alive_components] / (thermals[alive_components] / 100)) - steps_taken + 1

        return int(np.ceil(np.amin(timesteps)))

    def step(self, alive_components, thermals):
        """ Increment a timestep regarding the aging process of the simulation

        :param alive_components: 2D numpy boolean array indicating the position of alive components.
        :param thermals: 2D numpy float array with the current local thermals at this iteration.
        :return: Boolean - indicating if any new failures have occurred (which should be handled).
        """
        return self.update_agings(alive_components, thermals)

    def do_n_steps(self, n, alive_components, thermals):
        self._cur_agings[alive_components] += (n + 1) * self._lambdas[alive_components] * \
                                              (thermals[alive_components] / 100)

        assert np.any((self._cur_agings >= 1.0)[alive_components]), "n steps did not result in aging > 1.0" + \
                                                                    str(self._cur_agings[alive_components])

        return np.any((self._cur_agings >= 1.0)[alive_components])
