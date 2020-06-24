#!/usr/bin/env python

""" Provides all functionality of the chromosome aspects regarding the application mapping
of a design point.

Consists of a application mapping function, which will be used to actually create the app_mapping.
"""

import random

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"

from design.mapping import InvalidMappingError


class AppMapping:
    def __init__(self, map_func):
        """ Initialization of a genetic AppMapping object.

        :param map_func: mapping function to be used to create the
                         initial application mapping
        """
        self.map_func = map_func

    def __repr__(self):
        """ String representation of a AppMapping object.

        :return: string - representation of this object.
        """
        return str(self.map_func.__name__)

    def is_valid(self, caps, apps):
        try:
            self.map_func(caps, apps)
            return True
        except InvalidMappingError:
            return False

    def mutate(self, search_space):
        """ Mutate this AppMapping by altering its mapping strategy by replacing it
        with a random other application mapping algorithm (as defined in the search_space).

        :param search_space: SearchSpace object.
        :return: None
        """
        self.map_func = random.choice(search_space.map_strats)

    @staticmethod
    def mate(parent1, parent2):
        """ Uses the parents application mapping to determine the AppMapping
        gene for the children

        :param parent1: AppMapping (genetic) object
        :param parent2: AppMapping (genetic) object
        :return: AppMapping (genetic) child1, child2
        """
        return parent1, parent2
