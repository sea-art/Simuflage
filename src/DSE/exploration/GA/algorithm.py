#!/usr/bin/env python

""" Contains all overarching functionality regarding the genetic algorithm of the DSE.
"""
import logging
import random

import numpy as np
from deap import creator, base, tools
from deap.tools import sortNondominated

from DSE.evaluation import monte_carlo
from DSE.evaluation.Pareto_UCB1 import pareto_ucb1
from DSE.evaluation.SAR import sSAR
from DSE.evaluation.evaluation_tests import scalarized_lambda, get_all_weights
from DSE.exploration.GA import Chromosome, SearchSpace

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"

from DSE.exploration.GA.ga_logger import LoggerGA

CXPB = 0.5  # crossover probability
MUTPB = 0.3  # mutation probability
N_POP = 30
N_GENS = 50
REF_POINTS = tools.uniform_reference_points(3)


# Has to be defined globally
# https://stackoverflow.com/a/61082335
weights = (1.0, -1.0, -1.0)
creator.create("FitnessDSE", base.Fitness, weights=weights)
creator.create("Individual", Chromosome, fitness=creator.FitnessDSE)

S = [scalarized_lambda(w) for w in get_all_weights() if w[2] != 1.0]


class GA:
    def __init__(self, n_pop, n_gens, nr_samples, search_space, init_pop=None, eval_method='mcs', log_info=False, log_filename="out/default_log.csv"):
        """ Initialize a GA (Genetic Algorithm) object to run the GA.

        :param log_info: boolean to indicate if GA should log info
        :param eval_method: choice of ['mcs', 'ssar', 'pucb']
        :param n: integer - population size
        :param search_space: SearchSpace object
        """
        # print("~~~~ Initializing ~~~~")
        self.sesp = search_space
        self.eval_method = eval_method
        self.nr_samples = nr_samples

        self.logging = log_info

        if self.logging:
            self.logger = LoggerGA(log_filename, log_filename, logging.DEBUG)

        self.tb = self._init_toolbox()
        if init_pop is None:
            self.pop = self.tb.population(n_pop)
        else:
            self.pop = [creator.Individual(*chromosome.genes, search_space) for chromosome in init_pop]

        self.n_gens = n_gens
        self.generation = 0

    def _init_toolbox(self):
        """ Initialization of the DEAP toolbox containing all GA functions.

        :return: ToolBox object
        """
        # weights indicates whether to maximize or minimize the objectives.
        # Objectives are currently (MTTF, energy-consumption, size)
        # So weights right now indicate: maximize MTTF, minimize consumption+size.

        toolbox = base.Toolbox()
        toolbox.register("individual", Chromosome.create_random, creator.Individual, self.sesp)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("select", tools.selNSGA2)
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

        if self.eval_method == 'ssar':
            _, results, _ = sSAR(to_evaluate, len(to_evaluate) // 2, S, self.nr_samples)
        elif self.eval_method == 'pucb':
            results, _ = pareto_ucb1(to_evaluate, self.nr_samples)
        else:
            results = monte_carlo(to_evaluate, iterations=self.nr_samples, parallelized=False)

        for i in range(len(results)):
            to_evaluate[i].fitness.values = tuple(results[i])

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
        if self.logging:
            self.log_info()

        self.pop = self.tb.select(self.pop, N_POP)

    def _log_get_values(self):
        ttfs = []
        pes = []
        sizes = []

        for individual in self.pop:
            ttf, pe, size = individual.fitness.values

            ttfs.append(ttf)
            pes.append(pe)
            sizes.append(size)

        return ttfs, pes, sizes

    def log_info(self):
        gen_values = self._log_get_values()
        self.logger.log(self.generation, gen_values)

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

    def run(self):
        for _ in range(self.n_gens):
            self.next_generation()


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
    print("Starting GA with\npopulation: \t{}\ngenerations:\t {}\n~~~~~~~~~~~~~~~~~~~\n".format(N_POP, N_GENS))

    sesp = initialize_sesp()
    # ga = GA(N_POP, N_GENS, 1000, sesp, eval_method='mcs', log_filename="out/test.csv")
    init_pop = [Chromosome.create_random(Chromosome, sesp) for _ in range(5)]

    ga1 = GA(N_POP, N_GENS, 1000, sesp, init_pop=init_pop, eval_method='mcs')
    ga2 = GA(N_POP, N_GENS, 1000, sesp, init_pop=init_pop, eval_method='ssar')



if __name__ == "__main__":
    main()
