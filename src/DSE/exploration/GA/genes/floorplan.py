#!/usr/bin/env python

""" Provides all functionality of the chromosome aspects regarding the component's location
of a design point.

This file implements the genetic representation of locations and genetic algorithm operators
such as
- mutate
- crossover
- selection
"""

import numpy as np

from DSE.exploration.GA.operators import one_point_swapover
from design.mapping import all_possible_pos_mappings

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


class FloorPlan:
    def __init__(self, locations):
        """ Initialization of a genetic component FloorPlan (location) object.

        :param locations: list of integer tuples (x, y) indicating the locations of components.
        """
        self.locations = locations

    def __repr__(self):
        """ String representation of a FloorPlan object.

        :return: string - representation of this object.
        """
        return str(self.locations)

    def mutate(self, search_space):
        """ Mutate this FloorPlan object.

        Will randomly replace the location of a component with a random location
        that is not used by another components.

        :param search_space: SearchSpace object
        :return: None
        """
        a = all_possible_pos_mappings(search_space.max_components)
        b = np.asarray(self.locations)

        # https://stackoverflow.com/a/51352806
        ops = a[np.invert((a[:, None] == b).all(-1).any(-1))]
        idx = np.random.randint(len(self.locations))

        self.locations[idx] = tuple(ops[np.random.randint(ops.shape[0], size=1), :][0])

    @staticmethod
    def mate(parent1, parent2):
        """ One point swapover between two given parents.

        For details about the swapover function, see respective comment.

        :param parent1: FloorPlan (genetic) object
        :param parent2: FloorPlan (genetic) object
        :return: FloorPlan (genetic) child1 and child2
        """
        c1, c2 = one_point_swapover(parent1.locations, parent2.locations)

        return FloorPlan(c1), FloorPlan(c2)
