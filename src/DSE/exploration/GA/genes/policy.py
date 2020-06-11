#!/usr/bin/env python

""" Provides all functionality of the chromosome aspects regarding the adaptive policy
of a design point.

This file implements the genetic representation of the policy and genetic algorithm operators
such as
- mutate
- crossover
- selection
"""

import numpy as np

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


class Policy:
    def __init__(self, policy):
        """ Initialization of a genetic Policy object.

        :param policy: string indicating which policy to use.
        """
        self.policy = policy

    def __repr__(self):
        """ String representation of a Policy object.

        :return: string - representation of this object.
        """
        return self.policy

    def mutate(self, search_space):
        """ Mutate this Policy object.

        Will replace the current policy with a different policy out of the ones
        listed in the search space.

        :param search_space: SearchSpace object
        :return: None
        """
        self.policy = np.random.choice(search_space.policies[search_space.policies != self.policy])

    @staticmethod
    def mate(parent1, parent2):
        """ Swaps the policies between the two parents

        :param parent1: Policy (genetic) object
        :param parent2: Policy (genetic) object
        :return: Policy (genetic) child1 and child2
        """
        return Policy(parent1.policy), Policy(parent2.policy)
