import numpy as np
from deap.tools import cxOnePoint


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
        print(parent1.values)
        print(parent2.values)

        # TODO: unequal length
        c1, c2 = cxOnePoint(parent1.values, parent2.values)

        return Components(c1), Components(c2)
