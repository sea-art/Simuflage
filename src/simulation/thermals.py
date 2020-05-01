import numpy as np
from scipy import signal

from simulation.simulator_element import SimulatorElement


class Thermals(SimulatorElement):
    def __init__(self, init_thermals):
        """ Initializes a Thermals object based on the current initial thermals.

        :param init_thermals: 2D numpy float array containing the local temperatures.
        """
        self._isolated_thermals = init_thermals
        self._neighbour_thermals = np.zeros(init_thermals.shape)
        self._temps = np.copy(init_thermals)

    @property
    def temps(self):
        """ Getter function for the temps instance variable.

        :return: 2D float numpy array with temperatures on component positions.
        """
        return self._temps

    def adjusted_thermals(self, m, fluctuate):
        """ Adjusts the thermals based on uniform fluctuation and neighbour thermal influences.

        :param m: mapping of component index to xy-location (i, x , y)
        :param fluctuate: float representing the max uniformly fluctuation of temperature each iteration.
        :return: NOne
        """
        self._isolated_thermals[m['y'], m['x']] += np.random.uniform(-fluctuate, fluctuate,
                                                                    self._isolated_thermals.shape)[m['y'], m['x']]
        _neighbour_thermals = self.neighbour_thermal_influences()

        self._temps[m['y'], m['x']] = _neighbour_thermals[m['y'], m['x']]

    def neighbour_thermal_influences(self, kernel=None):
        """ Adjusts the thermals based on the neighbouring components thermals

        :param kernel: 2D kernel which will be used for convolution
        :return: 2D numpy float array - grid thermals after neighbouring contributions
        """
        if not kernel:
            kernel = np.asarray([[0.01, 0.01, 0.01],
                                 [0.01, 1, 0.01],
                                 [0.01, 0.01, 0.01]])

        return signal.convolve2d(self._isolated_thermals, kernel, "same")

    def step(self, comp_loc_map, fluctuate=0.01):
        """ Iterate the thermal influences

        :param comp_loc_map: (np structured array) mapping of component index to x, y location
        :param fluctuate: (float) representing the max uniformly fluctuation of temperature each iteration.
        :return: None
        """
        self.adjusted_thermals(comp_loc_map, fluctuate)
