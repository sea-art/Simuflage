#!/usr/bin/env python

""" Contains all overarching functionality regarding the genetic algorithm of the DSE.
"""
import itertools
import random
from copy import copy

import numpy as np
import scipy.stats as st
from deap.benchmarks.tools import hypervolume
from deap import creator, base, tools
from deap.tools import sortNondominated

from DSE.evaluation import monte_carlo
from DSE.evaluation.Pareto_UCB1 import pareto_ucb1
from DSE.evaluation.SAR import sSAR

from DSE.exploration import Chromosome, SearchSpace

REF_POINTS = tools.uniform_reference_points(3)


# Has to be defined globally
# https://stackoverflow.com/a/61082335
weights = (1.0, -1.0, -1.0)
creator.create("FitnessDSE", base.Fitness, weights=weights)
creator.create("Individual", Chromosome, fitness=creator.FitnessDSE)


def scalarized_lambda(w):
    """ Return function that will linearly scalarize a given vector based on weights w.
    :param w: iterable
    :return: lambda function vec: scalarized reward
    """
    return lambda vec: linear_scalarize(vec, w)


def linear_scalarize(vector, weights):
    """ Will linearly scalarize a given vector based on the given weights

    :param vector: iterable of n elements containing the rewards
    :param weights: iterable of n elements containing the weights of the rewards
    :return: float - single scalarized reward
    """
    return sum([vector[i] * weights[i] for i in range(len(vector))])


def get_all_weights(steps=10):
    """ Will create a list of all possible weight values for the scalarization functions.

    :type steps: int representing the stepsize (as 1 / steps)
    :return:
    """
    boxes = 3
    values = steps

    rng = list(range(values + 1)) * boxes
    ans = sorted(set(i for i in itertools.permutations(rng, boxes) if sum(i) == values))

    return list(map(lambda tup: (tup[0] / steps, -tup[1] / steps, -tup[2] / steps), ans))


S = [scalarized_lambda(w) for w in get_all_weights() if w[2] != -1.0][::7]


class GA:
    def __init__(self, n_pop, n_gens, samples_per_dp, search_space, init_pop=None, eval_method='mcs', mutpb=0.3, log=False):
        """ Initialize a GA (Genetic Algorithm) object to run the GA.

        :param init_pop: list of Individual objects
        :param log_info: boolean to indicate if GA should log info
        :param eval_method: choice of ['mcs', 'ssar', 'pucb']
        :param n: integer - population size
        :param search_space: SearchSpace object
        """
        self.n_pop = n_pop
        self.sesp = search_space
        self.eval_method = eval_method
        self.samples_per_dp = samples_per_dp
        self.mutpb = mutpb
        self.islogging = log

        self.ref_point = np.array([-1, 600.0, 37.0])

        self._nr_mutations = 0  # Used to log how many mutations occurred information
        self._death_penalty = 0  # Used to log how mutations resulted in death penalty
        self._nr_offspring = 0  # Used to log how many offspring was created
        self._samples_spent_per_dp = 0  # Used to log how many samples were spent this generation (mainly for ssar)

        self.tb = self._init_toolbox()
        self.stats = self._init_statistics()
        self.logbook = tools.Logbook()

        if init_pop is None:
            self.pop = self.tb.population(n_pop)
        else:
            # self.pop = [creator.Individual(*chromosome.genes, search_space) for chromosome in init_pop]
            self.pop = init_pop

        self.prev_pop = copy(self.pop)

        self.n_gens = n_gens
        self.generation = 0

        print("Generation:", self.generation)
        self.evaluate()
        if self.islogging:
            self.log()

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
        stats = tools.Statistics()
        stats.register("mean", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("best", lambda ind: np.array((np.max(ind, axis=0)[0],
                                                     np.min(ind, axis=0)[1],
                                                     np.min(ind, axis=0)[2])))
        stats.register("median", np.median, axis=0)
        stats.register("skew", st.skew, axis=0)

        return stats

    def _set_fitness_values(self, values):
        assert len(self.pop) == len(values)

        for ind, rew in zip(self.pop, values):
            ind.fitness.values = rew

    def _calc_distance(self):
        aprox_front = sortNondominated(self.pop, 10, first_front_only=True)[0]

        return hypervolume(aprox_front, self.ref_point)

    def log(self):
        real_data = np.array(self._mcs())

        actual_record_stats = self.stats.compile(real_data)
        own_record_stats = {'sampled_' + k: v for k, v in
                            self.stats.compile(np.array([k.fitness.values for k in self.pop])).items()}

        operator_stats = {'mutations': self._nr_mutations, 'death_penalty': self._death_penalty,
                          'offspring': self._nr_offspring, 'elitism': len((set(self.pop) & set(self.prev_pop))),
                          'sampleddistance': self._calc_distance(), "spent_samples": self._samples_spent_per_dp}

        sampled_fitness = [ind.fitness.values for ind in self.pop]
        self._set_fitness_values(real_data)
        operator_stats['distance'] = self._calc_distance()
        self._set_fitness_values(sampled_fitness)

        self.logbook.record(gen=self.generation, **actual_record_stats, **operator_stats, ** own_record_stats)

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
            _, results, N = sSAR(to_evaluate, len(to_evaluate) // 2, S, self.samples_per_dp * len(to_evaluate))
            N = sum(N) / len(to_evaluate)
        elif self.eval_method == 'pucb':
            results, N = pareto_ucb1(to_evaluate, len(to_evaluate) // 2, samples)
            N = sum(N) / len(to_evaluate)
        else:
            results = monte_carlo(to_evaluate, sample_budget=samples, parallelized=False)
            N = samples

        self._samples_spent_per_dp = N
        print("samples spent:", N)

        for i in range(len(results)):
            to_evaluate[i].fitness.values = tuple(results[i])

    def _mcs(self):
        self._ref_nr_samples = 100

        results = list(monte_carlo(self.pop, sample_budget=self._ref_nr_samples * len(self.pop),
                                   parallelized=False).values())

        return results

    def crossover(self):
        """ Separates the population in two and crosses the individuals based on probability.

        :return: offspring
        """
        offspring = []

        for parent1, parent2 in zip(self.pop[::2], self.pop[1::2]):
            offspring += list(Chromosome.mate(parent1, parent2, self.sesp))

        self._death_penalty += len(offspring)

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

        if self.islogging:
            self.log()

    def next_generation(self):
        """ Main loop per generation.

        Will crossover the current individuals, mutate them and then select
        individuals for the new generation.

        :return: None
        """
        self.prev_pop = self.pop
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
    sesp = initialize_sesp()

    ga = GA(100, 5, 1, sesp)
    ga.run()


if __name__ == "__main__":
    main()
