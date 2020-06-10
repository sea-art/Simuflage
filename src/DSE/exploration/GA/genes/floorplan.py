from deap.tools import cxOrdered

from DSE.exploration.GA.operators import one_point_crossover
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
        c1, c2 = one_point_crossover(parent1.locations, parent2.locations)

        c1 = list(map(tuple, c1.reshape((len(c1) // 2, 2))))
        c2 = list(map(tuple, c2.reshape((len(c2) // 2, 2))))

        if len(c1) != len(set(c1)):
            FloorPlan.repair(c1)

        if len(c2) != len(set(c2)):
            FloorPlan.repair(c2)

        return FloorPlan(c1), FloorPlan(c2)

    @staticmethod
    def repair(c):
        pass
