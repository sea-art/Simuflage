#!/usr/bin/env python


""" Contains all evaluation methods regarding the k-armed bandit problem.
"""

import random

from DSE import monte_carlo
from simulation import Simulator


__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


def avg(lst):
    return sum(lst) / len(lst)


####################
# SINGLE OBJECTIVE #
####################
def mab_greedy(designpoints, nr_samples=10000, idx=0, func=max):
    """ Single objective greedy evaluation of the given list of design points.

    Will first explore all design points once and then only take greedy actions.
    Since the output of the simulator will have multiple values (multi-objective), the idx
    variable can be used to specify which value should be used by the MAB (default=TTF).

    :param designpoints: [DesignPoint object] - List of DesignPoint objects (the candidates).
    :param nr_samples: number of samples
    :param idx: index of simulator return value to use as objective
    :param func: function to select the best DesignPoint (should be max or min).
    :return: [(mean of samples, nr_samples)] - will return the mean of the sampled values
                                               and the amount of samples taken for this dp.
    """
    # Uses epsilon_greedy with an epsilon of 0.0 (no exploration)
    return mab_epsilon_greedy(designpoints, 0.0, nr_samples, idx=idx, func=func)


def mab_epsilon_greedy(designpoints, e, nr_samples=10000, idx=0, func=max):
    """ Single objective epsilon-greedy evaluation of the given list of design points.

    Will first explore all design points once and then only take epsilon-greedy actions.
    Since the output of the simulator will have multiple values (multi-objective), the idx
    variable can be used to specify which value should be used by the MAB (default=TTF).

    :param designpoints: [DesignPoint object] - List of DesignPoint objects (the candidates).
    :param e: epsilon value between [0.0 - 1.0] indicating the probability to explore a random dp.
    :param nr_samples: number of samples
    :param idx: index of simulator return value to use as objective
    :param func: function to select the best DesignPoint (should be max or min).
    :return: [(mean of samples, nr_samples)] - will return the mean of the sampled values
                                               and the amount of samples taken for this dp.
    """
    simulators = [Simulator(dp) for dp in designpoints]
    samples = [[sim.run_optimized()[idx]] for sim in simulators]  # Explores all candidates once.
    averages = [x[0] for x in samples]

    for _ in range(len(simulators), nr_samples):
        if random.random() < e:  # epsilon exploration
            i = random.randint(0, len(simulators) - 1)
        else:  # greedy exploitation
            i = random.choice([i for i, val in enumerate(averages) if val == func(averages)])

        new_sample = simulators[i].run_optimized()[idx]
        samples[i].append(new_sample)
        averages[i] = avg(samples[i])

    return [(avg(x), len(x)) for x in samples]

