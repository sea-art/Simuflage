#!/usr/bin/env python

""" Provides all functionality of the chromosome aspects regarding the component's capacity
of a design point.

This file implements the genetic representation of components and genetic algorithm operators
such as
- mutate
- crossover
- selection
"""

import numpy as np

from DSE.exploration.GA.operators import one_point_crossover

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


class Components:
    def __init__(self, capacities):
        """ Initialization of a genetic component capacity object.

        :param capacities: list of integers representing the capacities of components/
        """
        self.values = capacities

    def __repr__(self):
        """ String representation of a Components object.

        :return: string - representation of this object.
        """
        return str(self.values)

    def mutate(self, search_space):
        """ Mutate this Components genetic object.

        Will randomly replace the capacity of a single component with a different capacity
        out of all possible capacities from the search space.

        :param search_space: SearchSpace object
        :return: None
        """
        # This index (idx) will be mutated by randomly selecting another component
        idx = np.random.randint(len(self.values))

        possible_values = search_space.capacities[search_space.capacities != self.values[idx]]
        self.values[idx] = np.random.choice(possible_values)

    @staticmethod
    def mate(parent1, parent2):
        """ One point crossover between two given parents.

        Does NOT adjust the parents in place, but returns adjusted children while leaving
        the parents intact.

        :param parent1: Components (genetic) object
        :param parent2: Components (genetic) object
        :return: tuple: Components (genetic) child1 and child2
        """
        c1, c2 = one_point_crossover(parent1.values, parent2.values)

        return Components(c1), Components(c2)
