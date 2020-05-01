import numpy as np
from scipy import signal


class Thermals:
    def __init__(self, init_thermals):
        """ Initializes a Thermals object based on the current initial thermals.

        :param init_thermals: 2D numpy float array containing the local temperatures.
        """
        self.isolated_thermals = init_thermals
        self.neighbour_thermals = np.zeros(init_thermals.shape)
        self.temps = np.copy(init_thermals)

    def adjusted_thermals(self, m, fluctuate):
        """ Adjusts the thermals based on uniform fluctuation and neighbour thermal influences.

        :param m: mapping of component index to xy-location (i, x , y)
        :param fluctuate: float representing the max uniformly fluctuation of temperature each iteration.
        :return: NOne
        """
        self.isolated_thermals[m['y'], m['x']] += np.random.uniform(-fluctuate, fluctuate,
                                                                    self.isolated_thermals.shape)[m['y'], m['x']]
        neighbour_thermals = self.neighbour_thermal_influences()

        self.temps[m['y'], m['x']] = neighbour_thermals[m['y'], m['x']]

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

    def step(self, comp_loc_map, fluctuate=0.01):
        """ Iterate the thermal influences

        :param comp_loc_map: (np structured array) mapping of component index to x, y location
        :param fluctuate: (float) representing the max uniformly fluctuation of temperature each iteration.
        :return: None
        """
        self.adjusted_thermals(comp_loc_map, fluctuate)
