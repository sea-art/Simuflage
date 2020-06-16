#!/usr/bin/env python

"Contains all overarching functionality regarding the genetic algorithm of the DSE."

import random
import numpy as np
from deap import creator, base, tools
from deap.tools import selNSGA3

from DSE.evaluation import monte_carlo
from DSE.exploration.GA import Chromosome, SearchSpace

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"

CXPB = 0.5  # crossover probability
MUTPB = 0.3  # mutation probability
N_POP = 200

REF_POINTS = tools.uniform_reference_points(3)


class GA:
    def __init__(self, n, search_space):
        """ Initialize a GA (Genetic Algorithm) object to run the GA.

        :param n: integer - population size
        :param search_space: SearchSpace object
        """
        self.sesp = search_space
        self.tb = self._init_toolbox()

        self.pop = self.tb.population(N_POP)

        self.generation = 0

        # Evaluates the initial population.
        self.evaluate()

    def _init_toolbox(self):
        """ Initialization of the DEAP toolbox containing all GA functions.

        :return: ToolBox object
        """
        # weights indicates whether to maximize or minimize the objectives.
        # Objectives are currently (MTTF, energy-consumption, size)
        # So weights right now indicate: maximize MTTF, minimize consumption+size.
        creator.create("FitnessDSE", base.Fitness, weights=(1.0, -1.0, -1.0))
        creator.create("Individual", Chromosome, fitness=creator.FitnessDSE)

        toolbox = base.Toolbox()
        toolbox.register("individual", Chromosome.create_random, creator.Individual, self.sesp)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("select", tools.selNSGA3WithMemory(REF_POINTS))
        toolbox.register("evaluate", self.evaluate)

        return toolbox

    def evaluate(self):
        """ Evaluate the current generation (self._generation) via monte carlo simulation.

        Only evaluates the individuals that have not yet been evaluated.

        :return: None
        """
        # All individuals that are not yet evaluated
        to_evaluate = [c for c in self.pop if not c.fitness.valid]

        # If there's nothing to evaluate, return
        if len(to_evaluate) == 0:
            return

        # results is a dict mapping index to tuple of n-values
        # e.g. {1: (102.7, 30.4), 2: (92,8, 60,2), ...}
        results = monte_carlo(to_evaluate, iterations=600, parallelized=False)

        for i in results:
            to_evaluate[i].fitness.values = results[i]

    def crossover(self):
        """ Separates the population in two and crosses the individuals based on probability.

        :return: offspring
        """
        offspring = []

        for parent1, parent2 in zip(self.pop[::2], self.pop[1::2]):
            if random.random() < CXPB:
                offspring += list(Chromosome.mate(parent1, parent2, self.sesp))

        return offspring

    def mutate(self, offspring):
        """ Mutates the individuals of the offspring based on the mutation probability.

        :return: None
        """
        for c in offspring:
            c.mutate()

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

        offspring = self.crossover()
        self.mutate(offspring)
        self.pop = ga.pop + list(offspring)

        self.evaluate()
        self.select()


def initialize_sesp():
    """ Initializes a SearchSpace based on hardcoded values.

    :return: SearchSpace object.
    """
    capacities = np.array([40, 80, 150])
    applications = [5, 5, 10, 10]
    max_components = 20
    policies = np.array(['most', 'least', 'random'])

    return SearchSpace(capacities, applications, max_components, policies)


if __name__ == "__main__":
    sesp = initialize_sesp()
    ga = GA(N_POP, sesp)

    for _ in range(100):
        ga.next_generation()

        print("Generation:", ga.generation)

    for x in selNSGA3(ga.pop, 5, REF_POINTS):
        print(x, x.fitness.values)

    # for c in ga.pop:)
