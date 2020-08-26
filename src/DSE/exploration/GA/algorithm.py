#!/usr/bin/env python

""" Contains all overarching functionality regarding the genetic algorithm of the DSE.
"""
import logging
import random
from copy import copy

import numpy as np
import scipy.stats as st
from scipy.spatial import distance
from deap import creator, base, tools
from deap.tools import sortNondominated, Statistics

from DSE.evaluation import monte_carlo
from DSE.evaluation.Pareto_UCB1 import pareto_ucb1
from DSE.evaluation.SAR import sSAR
from DSE.evaluation.evaluation_tests import scalarized_lambda, get_all_weights
from DSE.exploration.GA import Chromosome, SearchSpace

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"

from DSE.exploration.GA.ga_logger import LoggerGA
from experiments import AnalysisGA

REF_POINTS = tools.uniform_reference_points(3)


# Has to be defined globally
# https://stackoverflow.com/a/61082335
weights = (1.0, -1.0, -1.0)
creator.create("FitnessDSE", base.Fitness, weights=weights)
creator.create("Individual", Chromosome, fitness=creator.FitnessDSE)

S = [scalarized_lambda(w) for w in get_all_weights() if w[2] != 1.0]


class GA:
    def __init__(self, n_pop, n_gens, samples_per_dp, search_space, init_pop=None, eval_method='mcs', mutpb=0.3,
                 ref_set=None):
        """ Initialize a GA (Genetic Algorithm) object to run the GA.

        :param log_info: boolean to indicate if GA should log info
        :param eval_method: choice of ['mcs', 'ssar', 'pucb']
        :param n: integer - population size
        :param search_space: SearchSpace object
        """
        # print("~~~~ Initializing ~~~~")
        self.n_pop = n_pop
        self.sesp = search_space
        self.eval_method = eval_method
        self.samples_per_dp = samples_per_dp
        self.mutpb = mutpb

        if ref_set is not None:
            self.ref_set = ref_set
        else:
            self.ref_set = None

        self._nr_mutations = 0  # Used to log how many mutations occurred information
        self._death_penalty = 0  # Used to log how mutations resulted in death penalty
        self._nr_offspring = 0  # Used to log how many offspring was created

        self.tb = self._init_toolbox()
        self.stats = self._init_statistics()
        self.logbook = tools.Logbook()

        if init_pop is None:
            self.pop = self.tb.population(n_pop)
        else:
            self.pop = [creator.Individual(*chromosome.genes, search_space) for chromosome in init_pop]

        self.prev_pop = copy(self.pop)

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

    def _init_statistics(self):
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("mean", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("best", lambda ind: np.array((np.max(ind, axis=0)[0],
                                                     np.min(ind, axis=0)[1],
                                                     np.min(ind, axis=0)[2])))
        stats.register("median", np.median, axis=0)
        stats.register("skew", st.skew, axis=0)

        return stats

    def _calc_distance(self):
        aprox_front = sortNondominated(self.pop, 10, first_front_only=True)[0]
        front_vals = np.array([a.fitness.values for a in aprox_front])

        return np.mean(np.min(distance.cdist(self.ref_set, front_vals), axis=0))

    def log(self):
        record_stats = self.stats.compile(self.pop)
        operator_stats = {'mutations': self._nr_mutations,
                          'death_penalty': self._death_penalty,
                          'offspring': self._nr_offspring,
                          'elitism': len((set(self.pop) & set(self.prev_pop)))}

        if self.ref_set is not None:
            operator_stats['distance'] = self._calc_distance()

        self.logbook.record(gen=self.generation, **record_stats, **operator_stats)

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

        samples = self.samples_per_dp * len(to_evaluate)

        if self.eval_method == 'ssar':
            _, results, _ = sSAR(to_evaluate, samples // 2, S, self.samples_per_dp)
        elif self.eval_method == 'pucb':
            results, _ = pareto_ucb1(to_evaluate, samples)
        else:
            results = monte_carlo(to_evaluate, iterations=samples, parallelized=False)

        for i in range(len(results)):
            to_evaluate[i].fitness.values = tuple(results[i])

    def crossover(self):
        """ Separates the population in two and crosses the individuals based on probability.

        :return: offspring
        """
        offspring = []

        print(len(self.pop))

        for parent1, parent2 in zip(self.pop[::2], self.pop[1::2]):
            offspring += list(Chromosome.mate(parent1, parent2, self.sesp))

        self._death_penalty += len(offspring)

        assert (len(offspring) == self.n_pop)

        # Checks if invalid individuals were created.
        offspring = [o for o in offspring if o.is_valid()]

        self._death_penalty -= len(offspring)
        self._nr_offspring = len(offspring)

        return offspring

    def mutate(self, offspring):
        """ Mutates the individuals of the offspring based on the mutation probability.

        :return: None
        """
        for c in offspring:
            if random.random() < self.mutpb:
                self._nr_mutations += 1
                c.mutate()

            # Adding/removing components will result in incorrect location gene,
            # which has to be repaired.
            if len(c.genes[0]) != len(c.genes[1]):
                c.genes[1].repair(c.genes[0], self.sesp)

        self._death_penalty += len(offspring)
        # Checks if mutated individuals are still valid.
        offspring = [o for o in offspring if o.is_valid()]
        self._death_penalty -= len(offspring)

        return offspring

    def select(self):
        """ Selects the best individuals of a population + offspring via NSGA-3.

        :return: population
        """
        self.pop = self.tb.select(self.pop, self.n_pop)
        self.log()

    # def _log_get_values(self):
    #     ttfs = []
    #     pes = []
    #     sizes = []
    #
    #     for individual in self.pop:
    #         ttf, pe, size = individual.fitness.values
    #
    #         ttfs.append(ttf)
    #         pes.append(pe)
    #         sizes.append(size)
    #
    #     return ttfs, pes, sizes


    def next_generation(self):
        """ Main loop per generation.

        Will crossover the current individuals, mutate them and then select
        individuals for the new generation.

        :return: None
        """
        self.prev_pop = self.pop[:]
        self._nr_mutations = 0
        self._death_penalty = 0
        self._nr_offspring = 0

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
    init_pop = [Chromosome.create_random(Chromosome, sesp) for _ in range(50)]

    ga1 = GA(N_POP, N_GENS, 1000, sesp, init_pop=init_pop, eval_method='mcs')

    ga1.run()
    print("best: ", sortNondominated(ga1.pop, 10, first_front_only=True)[0])

    # ga2 = GA(N_POP, N_GENS, 1000, sesp, init_pop=init_pop, eval_method='ssar')


if __name__ == "__main__":
    main()
