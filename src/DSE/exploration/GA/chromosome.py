from design import DesignPoint
from .genes import Components, FloorPlan, AppMapping, Policy

import numpy as np
import random

class Chromosome:
    def __init__(self, caps, locs, apps, maps, policy, search_space):
        self.genes = [Components(caps), FloorPlan(locs), AppMapping(maps), Policy(policy)]
        self.search_space = search_space

        # Actual DesignPoint object of this chromosome
        self.dp = DesignPoint.create(caps, locs, apps, maps, policy=policy)

    def __repr__(self):
        return "gene1: {}\ngene2: {}\ngene3: {}\ngene4: '{}'"\
            .format(self.genes[0], self.genes[1], self.genes[2], self.genes[3])

    @classmethod
    def create_random(cls, search_space):
        """

        :param search_space: SearchSpace object providing the degrees of freedom
        :return:
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
        """ Defines the

        :param c2:
        :return:
        """

        # defines randomly how many and which clusters will be mutated
        n_cluster_mutations = np.random.randint(len(self.genes))
        clusters_to_mutate = random.sample(self.genes, n_cluster_mutations)

        for cluster in clusters_to_mutate:
            cluster.mutate(self.search_space)
