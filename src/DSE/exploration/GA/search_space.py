#!/usr/bin/env python

""" SearchSpace object will provide the degrees of freedom required for the GA to get the
context of what to explore.
"""

from design.mapping import all_possible_pos_mappings

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


class SearchSpace:
    def __init__(self, capacities, applications, max_components, policies):
        """ Defines the search space by describing all possible dimensions of freedom within the search space.
        The given values will be stored in this object to be easily utilize the degrees of freedom.

        :param capacities: list of floats - possible capacities for components
        :param applications: list of floats - the applications that have to be executed
        :param max_components: integer indicating the upper bound of component selection
        :param policies - list of strings defining the possible policies
        """
        self.capacities = capacities
        self.loc_choices = list(map(tuple, all_possible_pos_mappings(26)))
        self.max_components = max_components

        self.applications = applications
        self.n_apps = len(applications)

        self.policies = policies
