#!/usr/bin/env python

""" Provides all functionality of the chromosome aspects regarding the component's capacity
of a design point.

This file implements the genetic representation of components and genetic algorithm operators
such as
- mutate
- crossover
- selection
"""

import random
import numpy as np

from DSE.exploration.operators import one_point_crossover


class Components:
    def __init__(self, capacities):
        """ Initialization of a genetic component capacity object.

        :param capacities: int numpy array representing the capabilities of components
        """
        self.values = capacities

    def __repr__(self):
        """ String representation of a Components object.

        :return: string - representation of this object.
        """
        return str(self.values)

    def __len__(self):
        return len(self.values)

    def mutate(self, search_space):
        """ Mutate this Components genetic object.

        Will do either one of two things:
        1) remove or add a random component
        2) Will randomly replace the capacity of a single component with a different capacity
        out of all possible capabilities from the search space.

        :param search_space: SearchSpace object
        :return: None
        """
        add_or_change = bool(random.getrandbits(1))

        if add_or_change:  # will add/remove component
            add_component = bool(random.getrandbits(1))

            if add_component or len(self.values) == 1:
                self.values = np.append(self.values, np.random.choice(search_space.capacities))
            else:
                self.values = np.delete(self.values, np.random.randint(0, len(self.values)))

        else:  # will change the capacity of a component
            idx = np.random.randint(0, len(self.values))

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
