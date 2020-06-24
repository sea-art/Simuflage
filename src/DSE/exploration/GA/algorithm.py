#!/usr/bin/env python

"Contains all overarching functionality regarding the genetic algorithm of the DSE."
import collections
import random
import numpy as np
from deap import creator, base, tools
from deap.tools import selNSGA3

from DSE.evaluation import monte_carlo, mab_sar
from DSE.exploration.GA import Chromosome, SearchSpace

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"

CXPB = 0.5  # crossover probability
MUTPB = 0.3  # mutation probability
N_POP = 40
N_GENS = 50
SCALARIZED = False
REF_POINTS = tools.uniform_reference_points(3)


class GA:
    def __init__(self, n, search_space, scalarized=False):
        """ Initialize a GA (Genetic Algorithm) object to run the GA.

        :param n: integer - population size
        :param search_space: SearchSpace object
        """
        print("~~~~ Initializing ~~~~")
        self.sesp = search_space
        self.scalarized = scalarized

        self.tb = self._init_toolbox()

        self.pop = self.tb.population(n)

        self.generation = 0

        # Evaluates the initial population.
        self.evaluate()

    def _init_toolbox(self):
        """ Initialization of the DEAP toolbox containing all GA functions.

        :return: ToolBox object
        """
        if self.scalarized:
            weights = (1.0,)
        else:
            weights = (1.0, -1.0, -1.0)

        # weights indicates whether to maximize or minimize the objectives.
        # Objectives are currently (MTTF, energy-consumption, size)
        # So weights right now indicate: maximize MTTF, minimize consumption+size.
        creator.create("FitnessDSE", base.Fitness, weights=weights)
        creator.create("Individual", Chromosome, fitness=creator.FitnessDSE)

        toolbox = base.Toolbox()
        toolbox.register("individual", Chromosome.create_random, creator.Individual, self.sesp)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        if self.scalarized:
            toolbox.register("select", tools.selBest)
        else:
            toolbox.register("select", tools.selNSGA3WithMemory(REF_POINTS))
        toolbox.register("evaluate", self.evaluate)

        return toolbox

    def evaluate(self):
        """ Evaluate the current generation (self._generation) via monte carlo simulation.

        Only evaluates the individuals that have not yet been evaluated.

        :type use_mab: bool - will evalute via MAB instead of MCS
        :return: None
        """
        # All individuals that are not yet evaluated
        to_evaluate = [c for c in self.pop if not c.fitness.valid]

        # If there's nothing to evaluate, return
        if len(to_evaluate) == 0:
            return

        # results is a dict mapping index to tuple of n-values
        # e.g. {1: (102.7, 30.4, ...), 2: (92,8, 60,2, ...), ...}
        if self.scalarized:
            results = collections.OrderedDict(mab_sar(to_evaluate, len(to_evaluate) - 2))
        else:
            # Parallelized could be set to True when defining DEAP creator globally
            # https://stackoverflow.com/a/61082335
            results = monte_carlo(to_evaluate, iterations=800, parallelized=False)

        for i in results:
            if self.scalarized:
                # fitness values must be a tuple
                to_evaluate[i].fitness.values = (results[i],)
            else:
                to_evaluate[i].fitness.values = results[i]

    def crossover(self):
        """ Separates the population in two and crosses the individuals based on probability.

        :return: offspring
        """
        offspring = []

        for parent1, parent2 in zip(self.pop[::2], self.pop[1::2]):
            if random.random() < CXPB:
                offspring += list(Chromosome.mate(parent1, parent2, self.sesp))

        # Checks if invalid individuals were created.
        offspring = [o for o in offspring if o.is_valid()]

        return offspring

    def mutate(self, offspring):
        """ Mutates the individuals of the offspring based on the mutation probability.

        :return: None
        """
        for c in offspring:
            if random.random() < MUTPB:
                c.mutate()

            # Adding/removing components will result in incorrect location gene,
            # which has to be repaired.
            if len(c.genes[0]) != len(c.genes[1]):
                c.genes[1].repair(c.genes[0], self.sesp)

        # Checks if mutated individuals are still valid.
        offspring = [o for o in offspring if o.is_valid()]

        return offspring

    def select(self):
        """ Selects the best individuals of a population + offspring via NSGA-3.

        :return: population
        """
        self.pop = self.tb.select(self.pop, N_POP)

    def next_generation(self):
        """ Main loop per generation.

        Will crossover the current individuals, mutate them and then select
        individuals for the new generation.

        :return: None
        """
        self.generation += 1
        print("Generation:", self.generation)

        offspring = self.crossover()
        offspring = self.mutate(offspring)
        self.pop = self.pop + list(offspring)

        self.evaluate()
        self.select()


def initialize_sesp():
    """ Initializes a SearchSpace based on hardcoded values.

    :return: SearchSpace object.
    """
    capacities = np.array([40, 80, 150])
    applications = [20, 20, 20, 20]
    max_components = 20
    policies = np.array(['most', 'least', 'random'])

    return SearchSpace(capacities, applications, max_components, policies)


def main():
    print("Starting GA with\npopulation:\t{}\ngenerations:\t{}\n~~~~~~~~~~~~~~~~~~~\n".format(N_POP, N_GENS))

    sesp = initialize_sesp()
    ga = GA(N_POP, sesp, scalarized=SCALARIZED)

    for _ in range(N_GENS):
        ga.next_generation()

    print("~~~~ Finding best candidates ~~~~")

    if ga.scalarized:
        for x in tools.selBest(ga.pop, N_POP):
            print(x, x.fitness.values)
    else:
        for x in selNSGA3(ga.pop, N_POP, REF_POINTS):
            print(x, x.fitness.values)


if __name__ == "__main__":
    main()
