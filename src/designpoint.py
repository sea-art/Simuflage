import numpy as np


class Designpoint:
    components = []
    applications = []
    application_map = {}

    def __init__(self, comp_list, app_list, app_map):
        """ Initialize a Design point.

        :param comp_list: list of Component objects
        :param app_list: list of Application objects
        :param app_map: dictionary mapping {Component : Appliction}
        """
        self.components = comp_list
        self.applications = app_list
        self.application_map = app_map

    def to_numpy(self):
        """Return the components of a designpoint as numpy array."""
        capacities = []
        temperatures = []
        power_uses = []

        for c in self.components:
            capacities.append(c.capacity)
            temperatures.append(c.base_temp)
            power_uses.append(c.power_used)

        return \
            np.asarray(capacities),     \
            np.asarray(temperatures),   \
            np.asarray(power_uses),     \
            np.asarray([(self.components.index(k), self.applications.index(self.application_map[k])) for k in self.application_map],
                       dtype=[('comp', 'i4'), ('app', 'i4')])  # dtypes are numpy indices
