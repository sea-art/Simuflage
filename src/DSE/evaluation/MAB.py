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


def compare_mabs(designpoints, nr_samples=2000, idx=0, func=max):
    """ Comparison function between different mab algorithms.

    NOTE: for testing purposes only

    :param designpoints: [DesignPoint object] - List of DesignPoint objects (the candidates).
    :param nr_samples: number of samples
    :param idx: index of simulator return value to use as objective
    :param func: function to select the best DesignPoint (should be max or min).
    :return: None
    """
    qt = monte_carlo(designpoints, iterations=nr_samples, parallelized=False).values()

    values = list(zip(
                  mab_so_epsilon_greedy(designpoints, 0.1, nr_samples=nr_samples, idx=idx, func=func),
                  mab_so_ucb(designpoints, 3, nr_samples=nr_samples, idx=idx, func=func),
                  mab_so_gradient(designpoints, 0.1, nr_samples=nr_samples, idx=idx),
                  mab_so_gape_v(designpoints, 0.08, 1000000, TOP_CANDIDATES, nr_samples=nr_samples, idx=idx),
                  mab_sar(designpoints, TOP_CANDIDATES, nr_samples=nr_samples, idx=idx),
                  [(x[0], nr_samples // len(designpoints)) for x in qt]))

    qt = monte_carlo(designpoints, iterations=nr_samples, parallelized=False).values()
    print("Best MCS candidates:", sorted(set([i for (i, val) in sorted(zip(np.arange(len(qt)), qt), key=lambda x: x[1], reverse=True)][:TOP_CANDIDATES])))

    labels = ["e-greedy:\t", "UCB:\t\t", "gradient:\t", "GapE-v:\t\t", "SAR:\t\t", "MCS:\t\t"]

    for i, u in enumerate(values):
        print("Design point", i)
        for i in range(len(labels)):
            print(labels[i], round(u[i][0] / 100000, 2), int(u[i][1]))

        print("")

    vals = [[] for _ in labels]

    for res in values:
        for i in range(len(res)):
            vals[i].append(res[i][0] * res[i][1])

    for i in range(len(vals)):
        vals[i] = sum(vals[i]) / 100000000

    print(vals)


###################
# MULTI OBJECTIVE #
###################

NR_OBJECTIVES = 3


def add_confidence_interval(individuals, N, A_star_len, subtract=False):
    """ Adds a confidence interval to the current fitness values of individuals

    :param individuals: Individual (DEAP) object
    :param N: {individual: sample_nr} - determines how many samples have been spent towards a individual
    :param A_star_len: int - length of A_star as in Pareto_UCB1 algorithm
    :param subtract: Boolean - reverses this function to subtract rather than add.
    :return: None
    """
    n = len(individuals)
    ci = [math.sqrt(2 * math.log(n * (NR_OBJECTIVES * A_star_len) ** (1 / 4)) / N[i]) for i in range(n)]

    for i in range(n):
        a, b, c = individuals[i].fitness.values
        if not subtract:
            individuals[i].fitness.values = (a + ci[i], b + ci[i], c + ci[i])
        else:
            individuals[i].fitness.values = (a - ci[i], b - ci[i], c - ci[i])


def pareto_ucb1(individuals, k, nr_samples=500):
    """ Implementation of the pareto Upper-Confidence-Bound1 (pareto UCB1) pseudocode [Drugan&Nowe(2013)]

    :param individuals: individuals provided by the ga (must have fitness attributes)
    :param k: The number of individuals to select.
    :return: [(mean of samples, nr_samples)] - will return the mean of the sampled values
                                               and the amount of samples taken for this dp.
    :return: [(idx, sampled_value)]
    """
    n = len(individuals)
    simulators = {individuals[i]: Simulator(individuals[i]) for i in range(n)}
    N = {individuals[i]: 1 for i in range(n)}

    # individual fitness values are empirical means
    for i, indiv in enumerate(individuals):
        mttf, pow_usage = simulators[indiv].run_optimized()
        size = individuals[i].evaluate_size()
        individuals[i].fitness.values = (mttf, pow_usage, size)

    samples = len(individuals)

    while samples < nr_samples:
        A_star = sortNondominated(individuals, k, first_front_only=True)[0]
        add_confidence_interval(individuals, list(N.values()), len(A_star))

        A_p = sortNondominated(individuals, k, first_front_only=True)[0]
        add_confidence_interval(individuals, list(N.values()), len(A_star), subtract=True)

        a = np.random.choice(A_p)

        N[a] += 1
        samples += 1

        old_mttf, old_usage, old_size = a.fitness.values
        mttf, usage = simulators[a].run_optimized()

        a.fitness.values = (old_mttf + (mttf - old_mttf) / N[a],
                            old_usage + (usage - old_usage) / N[a],
                            old_size)

    return list(zip([i.fitness.values for i in individuals], list(N.values())))


def n_k(k, n, D, log_d):
    if k == 0:
        return 0

    return math.ceil((1 / log_d) * ((n - D) / (D + 1 - k)))


def mab_sar(individuals, m, nr_samples=1000):
    """ Scalarized multi-objective MAB evaluation via Successive Accept Reject (SAR).

    :param individuals: individuals provided by the ga (must have fitness attributes)
    :param m: int - length of best-arm set
    :param nr_samples: int - amount of samples
    :return: [(idx, sampled_value)]
    """
    simulators = [Simulator(d) for d in individuals]
    D = len(individuals)
    A = [i for i in range(D)]
    N = [0 for _ in range(D)]
    ui = [(i, 0) for i in range(len(simulators))]  # empirical means
    S = set()

    LOG_D = 1 / 2 + sum([1 / i for i in range(2, D + 1)])
    m_o = m

    for k in range(1, D):
        samples = int(n_k(k, nr_samples, D, LOG_D) - n_k(k - 1, nr_samples, D, LOG_D))
        for i in A:
            for _ in range(samples):
                new_sample = l_scale(list(simulators[i].run_optimized()) + [individuals[i].evaluate_size()],
                                     weights=(1, -1, -1))
                N[i] += 1
                ui[i] = (i, ui[i][1] + (new_sample - ui[i][1]) / N[i])

        sorted_indices = [i for (i, val) in sorted(ui, key=lambda x: x[1], reverse=True) if i in A]

        i_star_up = sorted_indices[m_o]
        i_star_down = sorted_indices[m_o + 1]
        gap_d = [0 for _ in range(D)]

        for i in sorted_indices[:m_o]:
            gap_d[i] = ui[i][1] - ui[i_star_down][1]

        for i in sorted_indices[m_o:]:
            gap_d[i] = ui[i_star_up][1] - ui[i][1]

        j = gap_d.index(max(gap_d))

        if j == sorted_indices[0]:
            S.add(j)
            m_o -= 1

        A = [i for i in A if i != j]

    return ui


def l_scale(values, weights):
    """ Lineairly scalarizes the evaluation of a design point

    :param values: tuple of 3 elements (floats)
    :param weights: [float] - weights used for the scalarization
    :return: float - transferred multi-objective sampled values to a single value
    """
    mttf, usage, size = tuple(values)

    # Normalization by dividing via the mean (of 50000 random design points)
    mttf /= 429320.87498
    usage /= 235.9094298110201
    size /= 30.2112

    return mttf * weights[0] + usage * weights[1] + size * weights[2]
