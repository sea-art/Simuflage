import numpy as np


class Agings:
    """ Contains all logical operators based on the aging of components."""
    def __init__(self, alive_components):
        """ Initializes an aging grid (which computes the aging rate for each of the components) based on the Weibull
        distribution.

        :param alive_components: 2D numpy boolean array (True indicates a living component on that position)
        """
        omegas = np.zeros(alive_components.shape)
        omegas[alive_components] = 100 * np.random.weibull(5, np.sum(alive_components)) # repr. iterations on 100% usage when cpu will fail

        self.lambdas = np.divide(1, omegas, out=np.zeros_like(omegas), where=omegas != 0)
        self.cur_agings = np.zeros(alive_components.shape, dtype=np.float)  # Will increment each iteration

    def update_agings(self, alive_components, thermals):
        """ Update the aging values of all components with a single iteration.

        :return: None
        """

        assert np.all((self.lambdas * thermals / 100)[alive_components] >= 0), "Negative aging rate"

        self.cur_agings[alive_components] += (self.lambdas * thermals / 100)[alive_components]

    def iterate(self, alive_components, thermals):
        """ Run one iteration regarding the aging process of the simulation

        :param alive_components: 2D numpy boolean array indicating the position of alive components.
        :param thermals: 2D numpy float array with the current local thermals at this iteration.
        :return: None
        """
        self.update_agings(alive_components, thermals)
