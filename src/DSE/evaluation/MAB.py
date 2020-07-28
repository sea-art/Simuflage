#!/usr/bin/env python

""" Contains all evaluation methods regarding the k-armed bandit problem.
"""

import random
import math
import numpy as np

from deap.tools import sortNondominated

from DSE.evaluation import monte_carlo
from simulation import Simulator


__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


TOP_CANDIDATES = 5


####################
# SINGLE OBJECTIVE #
####################

def mab_so_greedy(designpoints, nr_samples=10000, idx=0, func=max):
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
    return mab_so_epsilon_greedy(designpoints, 0.0, nr_samples, idx=idx, func=func)


def mab_so_epsilon_greedy(designpoints, e, nr_samples=10000, idx=0, func=max):
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
    samples = [1 for _ in simulators]
    qt = [sim.run_optimized()[idx] for sim in simulators]  # Q_t = est value of action a at timestep t

    for _ in range(len(simulators), nr_samples):
        if random.random() < e:  # epsilon exploration
            a = random.randint(0, len(simulators) - 1)
        else:  # greedy exploitation
            a = random.choice([i for i, val in enumerate(qt) if val == func(qt)])

        new_sample = simulators[a].run_optimized()[idx]
        samples[a] += 1
        # Incremental implementation of MAB [Sutton & Barto(2011)].
        qt[a] += (new_sample - qt[a]) / samples[a]

    print("Best e-greedy candidates:", sorted(set([i for (i, val) in sorted(zip(np.arange(len(qt)), qt), key=lambda x: x[1], reverse=True)][:TOP_CANDIDATES])))

    return list(zip(qt, samples))


def mab_so_ucb(designpoints, c, nr_samples=10000, idx=0, func=max):
    """ Single objective Upper-Confidence-Bound (UCB) action selection

    Will first explore all design points once and then only take Upper-Confidence-Bound actions.
    Since the output of the simulator will have multiple values (multi-objective), the idx
    variable can be used to specify which value should be used by the MAB (default=TTF).

    :param designpoints: [DesignPoint object] - List of DesignPoint objects (the candidates).
    :param c: degree of exploration
    :param nr_samples: number of samples
    :param idx: index of simulator return value to use as objective
    :param func: function to select the best DesignPoint (should be max or min).
    :return: [(mean of samples, nr_samples)] - will return the mean of the sampled values
                                               and the amount of samples taken for this dp.
    """
    simulators = [Simulator(dp) for dp in designpoints]
    samples = [1 for _ in simulators]
    qt = [sim.run_optimized()[idx] for sim in simulators]  # Q_t = est value of action a at timestep t

    for t in range(len(simulators), nr_samples):
        actions = [qt[i] + c * math.sqrt(math.log(t) / samples[i]) for i in range(len(simulators))]
        a = random.choice([i for i, val in enumerate(actions) if val == func(actions)])
        samples[a] += 1
        new_sample = simulators[a].run_optimized()[idx]
        qt[a] += (new_sample - qt[a]) / samples[a]

    print("Best UCB candidates:", sorted(set([i for (i, val) in sorted(zip(np.arange(len(qt)), qt), key=lambda x: x[1], reverse=True)][:TOP_CANDIDATES])))

    return list(zip(qt, samples))


def mab_so_gradient(designpoints, step_size, nr_samples=10000, idx=0, func=max):
    """ Single objective gradient bandits algorithm.

    Will first explore all design points once and then only take Upper-Confidence-Bound actions.
    Since the output of the simulator will have multiple values (multi-objective), the idx
    variable can be used to specify which value should be used by the MAB (default=TTF).

    :param designpoints: [DesignPoint object] - List of DesignPoint objects (the candidates).
    :param step_size: step size of the algorithm TODO
    :param nr_samples: number of samples
    :param idx: index of simulator return value to use as objective
    :param func: function to select the best DesignPoint (should be max or min).
    :return: [(mean of samples, nr_samples)] - will return the mean of the sampled values
                                               and the amount of samples taken for this dp.
    """
    simulators = [Simulator(dp) for dp in designpoints]

    k = len(designpoints)  # total amount of bandits
    H = np.zeros(k)  # H from formula
    P = np.ones(k) / k  # probability per index
    N = np.zeros(k)  # amount of samples per index?
    qt = np.zeros(k)

    avg_reward = 0

    for t in range(1, nr_samples + 1):
        A = np.random.choice(k, 1, p=P)[0]
        N[A] += 1
        R = simulators[A].run_optimized()[idx]
        qt[A] += (R - qt[A]) / N[A]
        R /= 100000

        avg_reward += (R - avg_reward) / t
        baseline = avg_reward

        H[A] += step_size * (R - baseline) * (1 - P[A])

        for a in range(k):
            if a != A:
                H[a] -= step_size * (R - baseline) * P[a]

        aux_exp = np.exp(H)
        P = aux_exp / np.sum(aux_exp)

    print("Best Gradient candidates:", sorted(set([i for (i, val) in sorted(zip(np.arange(qt.size), qt), key=lambda x: x[1], reverse=True)][:TOP_CANDIDATES])))

    return list(zip(qt, N))


def mab_so_gape_v(designpoints, a, b, m, nr_samples=1000, idx=0):
    """ Single-objective MAB Gap-based Exploration with Variance (GapE-V).

    :param designpoints: [DesignPoint object] - List of DesignPoint objects (the candidates).
    :param a: degree of exploration (TODO: ?)
    :param b: float - maximum expected value from samples
    :param m: amount of designs to select
    :param nr_samples: number of samples
    :param idx: index of simulator return value to use as objective
    :return: [(mean of samples, nr_samples)] - will return the mean of the sampled values
                                               and the amount of samples taken for this dp.
    """
    simulators = [Simulator(dp) for dp in designpoints]
    ui = [(i, simulators[i].run_optimized()[idx]) for i in range(len(simulators))]  # empirical means
    oi = [0 for _ in range(len(simulators))]
    T = [1 for _ in range(len(designpoints))]

    gap_d = [0 for _ in range(len(designpoints))]
    indices = [0 for _ in range(len(designpoints))]

    for t in range(len(simulators), nr_samples):
        sorted_indices = [i for (i, val) in sorted(ui, key=lambda x: x[1], reverse=True)]

        i_star_up = sorted_indices[m]
        i_star_down = sorted_indices[m + 1]

        for i in sorted_indices[:m]:
            gap_d[i] = ui[i][1] - ui[i_star_down][1]

        for i in sorted_indices[m:]:
            gap_d[i] = ui[i_star_up][1] - ui[i][1]

        for i in sorted_indices:
            indices[i] = -gap_d[i] + math.sqrt((2 * a * oi[i]) / T[i]) + (7 * a * b) / (3 * T[i])

        j = indices.index(max(indices))
        new_sample = simulators[j].run_optimized()[idx]
        prev_ui = ui[j][1]  # stores the current empiric mean
        ui[j] = (j, ui[j][1] + (new_sample - ui[j][1]) / T[j])
        # iterative variance calculation for o_i
        oi[j] += ((new_sample - prev_ui) * (new_sample - ui[j][1]) - oi[j]) / T[j]
        T[j] += 1

    print("Best GapE-V candidates:", sorted(set([i for (i, val) in sorted(ui, key=lambda x: x[1], reverse=True)][:TOP_CANDIDATES])))

    return list(zip([x[1] for x in ui], T))
