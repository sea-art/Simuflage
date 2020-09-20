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

from DSE.exploration.operators import one_point_swapover
from design.mapping import all_possible_pos_mappings


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

    def __len__(self):
        """ Integer representing the length of the FloorPlan object

        :return: integer - representing the length
        """
        return len(self.locations)

    def _get_non_used_loc(self, search_space):
        a = all_possible_pos_mappings(search_space.max_components)
        b = np.asarray(self.locations)

        # https://stackoverflow.com/a/51352806
        ops = a[np.invert((a[:, None] == b).all(-1).any(-1))]
        return tuple(ops[np.random.randint(ops.shape[0], size=1), :][0])

    def mutate(self, search_space):
        """ Mutate this FloorPlan object.

        Will randomly replace the location of a component with a random location
        that is not used by another components.

        :param search_space: SearchSpace object
        :return: None
        """
        idx = np.random.randint(len(self.locations))
        loc = self._get_non_used_loc(search_space)

        self.locations[idx] = tuple(loc)

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

    def repair(self, components, search_space):
        """ Repairs the FloorPlan object by removing or adding a location.

        Floorplan is required to be corresponding with the components. When
        the components are mutated, it requires the FloorPlan to be repaired

        :param components: [float] - list of components
        :param search_space: SearchSpace object
        :return:
        """
        if len(components) > len(self):  # Component was added
            loc = self._get_non_used_loc(search_space)
            self.locations.append(loc)
        else:
            del self.locations[np.random.randint(0, len(self.locations))]
