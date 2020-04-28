import numpy as np
from scipy import signal


class Thermals:
    def __init__(self, init_thermals):
        self.isolated_thermals = init_thermals
        self.neighbour_thermals = np.zeros(init_thermals.shape)
        self.thermals = np.copy(init_thermals)

    def adjusted_thermals(self, m, fluctuate):
        """ Adjusts the thermals based on uniform fluctuation and neighbour thermal influences.

        :param m: mapping of component to location
        :param fluctuate: float representing the max uniformly fluctuation of temperature each iteration.
        :return: NOne
        """
        self.isolated_thermals[m['y'], m['x']] += np.random.uniform(-fluctuate, fluctuate,
                                                                    self.isolated_thermals.shape)[m['y'], m['x']]
        neighbour_thermals = self.neighbour_thermal_influences()

        self.thermals[m['y'], m['x']] = neighbour_thermals[m['y'], m['x']]

    def neighbour_thermal_influences(self, kernel=None):
        """ Adjusts the thermals based on the neighbouring components thermals

        :param kernel: 2D kernel which will be used for convolution
        :return: 2D numpy float array - grid thermals after neighbouring contributions
        """
        if not kernel:
            kernel = np.asarray([[0.01, 0.01, 0.01],
                                 [0.01, 1, 0.01],
                                 [0.01, 0.01, 0.01]])

        return signal.convolve2d(self.isolated_thermals, kernel, "same")

    def iterate(self, comp_loc_map, fluctuate=1.0):
        """ Iterate the thermal influences

        :param comp_loc_map: (np structured array) mapping of component index to x, y location
        :param fluctuate: (float) representing the max uniformly fluctuation of temperature each iteration.
        :return: (2D np float array) - adjusted thermals after iteration
        """
        self.adjusted_thermals(comp_loc_map, fluctuate)

        return self.thermals