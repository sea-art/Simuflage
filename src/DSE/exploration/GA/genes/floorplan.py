from deap.tools import cxOrdered

from design.mapping import all_possible_pos_mappings

import numpy as np


class FloorPlan:
    def __init__(self, locations):
        """

        :param locations: list of integer tuples (x, y) indicating the locations of components.
        """
        self.locations = locations

    def __repr__(self):
        return str(self.locations)

    def mutate(self, search_space):
        a = all_possible_pos_mappings(search_space.max_components)
        b = np.asarray(self.locations)

        # https://stackoverflow.com/a/51352806
        ops = a[np.invert((a[:, None] == b).all(-1).any(-1))]
        idx = np.random.randint(len(self.locations))

        self.locations[idx] = tuple(ops[np.random.randint(ops.shape[0], size=1), :][0])

    @staticmethod
    def mate(parent1, parent2):
        c1, c2 = cxOrdered(parent1.locations, parent2.locations)

        return FloorPlan(c1), FloorPlan(c2)