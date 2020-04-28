import numpy as np


class Agings:
    def __init__(self, alive_components):
        omegas = np.zeros(alive_components.shape)
        omegas[alive_components] = 100 * np.random.weibull(5, np.sum(alive_components)) # repr. iterations on 100% usage when cpu will fail

        self.lambdas = np.divide(1, omegas, out=np.zeros_like(omegas), where=omegas != 0)
        self.agings = np.zeros(alive_components.shape, dtype=np.float)  # Will increment each iteration

    def update_agings(self, alive_components, thermals):
        """ Update the aging values of all components with a single iteration.

        :return: None
        """
        self.agings[alive_components] += self.lambdas[alive_components] * (thermals / 100)

    def iterate(self, alive_components, thermals):
        self.update_agings(alive_components, thermals)
