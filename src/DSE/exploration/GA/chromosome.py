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
    def __init__(self, components, floor_plan, app_mapping, policy, search_space):
        """ Initialization of a design point chromosome.

        :param components: Components (genetic) object
        :param floor_plan: FloorPlan (genetic) object
        :param app_mapping: AppMapping (genetic) object
        :param policy: Policy (genetic) object
        :param search_space: SearchSpace object.
        """
        self.genes = [components, floor_plan, app_mapping, policy]
        self.search_space = search_space

    def __repr__(self):
        """ String representation of a Chromosome.

        :return: string - representation of this object.
        """
        return "gene1: {}\ngene2: {}\ngene3: {}\ngene4: '{}'\n"\
            .format(self.genes[0], self.genes[1], self.genes[2], self.genes[3])

    @staticmethod
    def create_random(container, search_space):
        """ Randomly creates a Chromosome (and thus design point).

        Randomly selects n components with random capacities and locations.
        Selects a random policy and randomly maps applications to the components.

        :param container: Chromosome object or wrapper object
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

        return container(Components(caps), FloorPlan(locs), AppMapping(maps), Policy(policy), search_space)

    def mutate(self):
        """ Mutate this Chromosome object.

        Randomly picks 1 to all gene 'clusters' to mutate.
        Cluster refers to an heterogeneous gene within the chromosome (e.g. FloorPlan, AppMapping, etc).

        :return: None
        """

        # defines randomly how many and which clusters will be mutated
        n_cluster_mutations = np.random.randint(1, len(self.genes))
        clusters_to_mutate = random.sample(self.genes, n_cluster_mutations)

        for cluster in clusters_to_mutate:
            cluster.mutate(self.search_space)

    @staticmethod
    def mate(parent1, parent2, sesp):
        """ Crossover between two given parent (Chromosomes).

        :param parent1: Chromosome object
        :param parent2: Chromosome object
        :param search_space: SearchSpace object
        :return: tuple of Chromosome objects - child1, child2
        """
        c1, c2 = Components.mate(parent1.genes[0], parent2.genes[0])
        f1, f2 = FloorPlan.mate(parent1.genes[1], parent2.genes[1])
        a1, a2 = AppMapping.mate(parent1.genes[2], parent2.genes[2])
        p1, p2 = Policy.mate(parent1.genes[3], parent2.genes[3])

        typ = type(parent1)

        return typ(c1, f1, a1, p1, sesp), typ(c2, f2, a2, p2, sesp)

    def evaluate_size(self):
        locs = self.genes[1].locations

        return (max(locs, key=lambda t: t[1])[1] - min(locs, key=lambda t: t[1])[1] + 1) * \
               (max(locs, key=lambda t: t[0])[0] - min(locs, key=lambda t: t[0])[0] + 1)

    def to_numpy(self):
        """ Create a DesignPoint object of this Chromosome and call the respective to_numpy function.

        :return: <see DesignPoint.to_numpy()>
        """
        caps = self.genes[0].values
        locs = self.genes[1].locations
        apps = self.search_space.applications
        maps = self.genes[2].app_mapping
        policy = self.genes[3].policy

        return DesignPoint.create(caps, locs, apps, maps, policy=policy).to_numpy()
