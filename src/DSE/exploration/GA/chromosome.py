#!/usr/bin/env python

""" Complete chromosome representation of a design point.
Contains all aspects of a designpoint, i.e.:
- Component capacities
- Component locations
- Application mapping
- Policy

Provides GA methods for chromosomes (e.g. mutation, crossover, selection)
"""

from design import DesignPoint
from .genes import Components, FloorPlan, AppMapping, Policy

import numpy as np
import random

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


class Chromosome:
    def __init__(self, caps, locs, apps, maps, policy, search_space):
        """ Initialization of a design point chromosome.

        :param caps: integer list - capacities
        :param locs: tuple list [(x, y)] - location of components
        :param apps: integer list - list of applications that have to be executed
        :param maps: tuple list [(c, a)] - mapping of components to applications
        :param policy: string - policy of use
        :param search_space: SearchSpace object.
        """
        self.genes = [Components(caps), FloorPlan(locs), AppMapping(maps), Policy(policy)]
        self.search_space = search_space

        # Actual DesignPoint object of this chromosome
        self.dp = DesignPoint.create(caps, locs, apps, maps, policy=policy)

    def __repr__(self):
        """ String representation of a Chromosome.

        :return: string - representation of this object.
        """
        return "gene1: {}\ngene2: {}\ngene3: {}\ngene4: '{}'"\
            .format(self.genes[0], self.genes[1], self.genes[2], self.genes[3])

    @classmethod
    def create_random(cls, search_space):
        """ Randomly creates a Chromosome (and thus design point).

        Randomly selects n components with random capacities and locations.
        Selects a random policy and randomly maps applications to the components.

        :param search_space: SearchSpace object providing the degrees of freedom
        :return: Chromosome object.
        """
        n_components = np.random.randint(1, search_space.max_components + 1)

        caps = np.random.choice(search_space.capacities, n_components)
        locs = random.sample(search_space.loc_choices, n_components)
        policy = np.random.choice(search_space.policies)

        # TODO: Needs verification if maps is correct (and act upon invalid solutions)
        # TODO: a) repair   b) death-penalty    c) other
        maps = [(np.random.randint(n_components), a) for a in range(search_space.n_apps)]

        return Chromosome(caps, locs, search_space.applications, maps, policy, search_space)

    def mutate(self):
        """ Mutates this Chromosome object.

        :return: None
        """

        # defines randomly how many and which clusters will be mutated
        n_cluster_mutations = np.random.randint(len(self.genes))
        clusters_to_mutate = random.sample(self.genes, n_cluster_mutations)

        for cluster in clusters_to_mutate:
            cluster.mutate(self.search_space)

    @staticmethod
    def mate(parent1, parent2):
        """ Crossover between two given parent (Chromosomes).

        :param parent1: Chromosome object
        :param parent2: Chromosome object
        :return: tuple of Chromosome objects - child1, child2
        """
        c1, c2 = Components.mate(parent1.genes[0], parent2.genes[0])
        f1, f2 = FloorPlan.mate(parent1.genes[1], parent2.genes[1])
        a1, a2 = AppMapping.mate(parent1.genes[2], parent2.genes[2])
        p1, p2 = Policy.mate(parent1.genes[3], parent2.genes[3])
