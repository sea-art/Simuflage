from copy import copy, deepcopy

import numpy as np

from DSE.exploration.GA.operators import one_point_crossover


class Components:
    def __init__(self, capacities):
        """

        :param capacities: list of integers representing the capacities of components/
        """
        self.values = capacities

    def __repr__(self):
        return str(self.values)

    def mutate(self, search_space):
        # This index (idx) will be mutated by randomly selecting another component
        idx = np.random.randint(len(self.values))

        possible_values = search_space.capacities[search_space.capacities != self.values[idx]]
        self.values[idx] = np.random.choice(possible_values)

    @staticmethod
    def mate(parent1, parent2):
        c1, c2 = one_point_crossover(parent1.values, parent2.values)

        return Components(c1), Components(c2)
