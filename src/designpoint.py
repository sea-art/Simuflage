import numpy as np


class Designpoint:
    components = []
    applications = []
    application_map = {}

    def __init__(self, comp_list, app_list, app_map):
        """ Initialize a Design point.

        :param comp_list: list of Component objects
        :param app_list: list of Application objects
        :param app_map: dictionary mapping {Component : [Appliction]}
        """
        self.components = comp_list
        self.applications = app_list
        self.application_map = app_map

    def to_numpy(self):
        """Return the components of a designpoint as numpy array."""
        capacities = []
        temperatures = []

        for c in self.components:
            capacities.append(c.capacity)
            temperatures.append(c.base_temp)

        return \
            np.asarray(capacities),                 \
            np.asarray(temperatures),               \
            self.calc_power_usage_per_component(),  \
            np.asarray([(self.components.index(comp), app.power_req) for comp, app in self.application_map],
                       dtype=[('comp', 'i4'), ('app', 'i4')])  # dtypes are numpy indices

    def calc_power_usage_per_component(self):
        foo = np.zeros(len(self.components))

        for a, b in self.application_map:
            foo[self.components.index(a)] += b.power_req

        return foo
