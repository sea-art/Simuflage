from design import DesignPoint
from .genes import Components, FloorPlan, AppMapping, Policy
from . import SearchSpace

import random


class Chromosome:
    def __init__(self, caps, locs, apps, maps, policy):
        self.gene1 = Components(caps)
        self.gene2 = FloorPlan(locs)
        self.gene3 = AppMapping(maps)
        self.gene4 = Policy(policy)

        # Actual DesignPoint object of this chromosome
        self.dp = DesignPoint.create(caps, locs, apps, maps, policy=policy)

    @staticmethod
    def create_random(search_space):
        """

        :param search_space: SearchSpace object providing the degrees of freedom
        :return:
        """
        n_components = random.randint(1, search_space.max_components)

        caps = random.choices(search_space.capacities, k=n_components)
        locs = random.sample(search_space.loc_choices, n_components)
        policy = random.choice(search_space.policies)

        # TODO: Needs verification if maps is correct (and act upon invalid solutions)
        # TODO: a) repair   b) death-penalty    c) other
        maps = [(random.randint(0, n_components - 1), a) for a in range(search_space.n_apps)]

        return Chromosome(caps, locs, search_space.applications, maps, policy)
