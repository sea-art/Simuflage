#!/usr/bin/env python

"Contains all overarching functionality regarding the genetic algorithm of the DSE."

import random
import numpy as np

from DSE.evaluation import monte_carlo
from DSE.exploration.GA import Chromosome, SearchSpace

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"

CXPB = 0.5  # crossover probability
MUTPB = 0.3  # mutation probability


class GA:
    def __init__(self, n, search_space):
        """ Initialize a GA (Genetic Algorithm) object to run the GA.

        :param n: integer - population size
        :param search_space: SearchSpace object
        """
        self._sesp = search_space
        self._population = np.array([Chromosome.create_random(search_space) for _ in range(n)])
        self._generation = 0

        # Evaluates the initial population.
        self.evaluate()

    def crossover(self):
        """ Separates the population in two and crosses the individuals based on probability.

        :return: None
        """
        offspring = []

        for parent1, parent2 in zip(self._population[::2], self._population[1::2]):
            if random.random() < CXPB:
                offspring += list(Chromosome.mate(parent1, parent2, self._sesp))

        self._population = np.append(self._population, np.array(offspring))

    def mutate(self):
        """ Mutates the individuals of the offspring based on the mutation probability.

        :return: None
        """
        probs = np.random.random(self._population.size)

        for c in self._population[probs < MUTPB]:
            c.mutate()
            c.values = None

    def evaluate(self):
        """ Evaluate the current generation (self._generation) with monte carlo simulations.

        Only evaluates the individuals that have not yet been evaluated (e.g. in a previous generation).

        :return: None
        """
        # All individuals that are not yet evaluated
        to_evaluate = [c for c in self._population if not c.valid]

        # If there's nothing to evaluate, return
        if len(to_evaluate) == 0:
            return

        # results is a dict mapping index to tuple of n-values
        # e.g. {1: (102.7, 30.4), 2: (92,8, 60,2), ...}
        results = monte_carlo(to_evaluate)

        for i in results:
            to_evaluate[i].values = results[i]

    def select(self):
        pass

    def next_generation(self):
        """ Main loop per generation.

        Will crossover the current individuals, mutate them and then select
        individuals for the new generation.

        :return: None
        """
        self._generation += 1

        self.crossover()
        self.mutate()
        self.evaluate()
        self.select()


def initialize_sesp():
    """ Initializes a SearchSpace based on hardcoded values.

    :return: SearchSpace object.
    """
    capacities = np.array([40, 80, 150])
    applications = [5, 5, 10, 10]
    max_components = 5
    policies = np.array(['most', 'least', 'random'])

    return SearchSpace(capacities, applications, max_components, policies)


if __name__ == "__main__":
    sesp = initialize_sesp()

    ga = GA(40, sesp)
    ga.crossover()
    ga.mutate()

    ga.evaluate()

    for z in ga._population:
        print(z.values)
