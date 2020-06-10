import numpy as np


class AppMapping:
    def __init__(self, app_mapping):
        """
        TODO: might change this list of tuples to actual string where index i means app i and value is comp_value.
        :param app_mapping: list of tuples mapping components to applications based on index [(comp, app)]
        """
        self.app_mapping = app_mapping

    def __repr__(self):
        return str(self.app_mapping)

    def mutate(self, search_space):
        idx = np.random.randint(len(self.app_mapping))

        possible_comps = np.arange(len(search_space.capacities))
        possible_comps = possible_comps[possible_comps != self.app_mapping[idx][0]]

        self.app_mapping[idx] = (np.random.choice(possible_comps), self.app_mapping[idx][1])
