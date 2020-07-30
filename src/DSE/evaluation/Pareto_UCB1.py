
from deap.tools import sortNondominated, selNSGA2
import math
import numpy as np

from DSE.evaluation.SAR import normalize
from simulation import Simulator

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
    ci = [math.sqrt(2 * math.log(sum(N) * (NR_OBJECTIVES * A_star_len) ** (1 / 4)) / N[i]) for i in range(n)]
    print("ci", list(zip(range(1, len(individuals)), ci)))

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
        mttf, pow_usage, size = normalize(simulators[indiv].run_optimized())
        # size = individuals[i].evaluate_size()
        individuals[i].fitness.values = (mttf, pow_usage, size)

    samples = len(individuals)

    while samples < nr_samples:
        # print("samples", samples)
        A_star = sortNondominated(individuals, k, first_front_only=True)[0]
        # A_star = selNSGA2(individuals, k)
        add_confidence_interval(individuals, list(N.values()), len(A_star))

        A_p = sortNondominated(individuals, k, first_front_only=True)[0]
        # A_p = selNSGA2(individuals, k)
        print(len(A_star), len(A_p))

        add_confidence_interval(individuals, list(N.values()), len(A_star), subtract=True)

        a = np.random.choice(A_p)

        N[a] += 1
        samples += 1

        old_mttf, old_usage, old_size = a.fitness.values
        mttf, usage, size = normalize(simulators[a].run_optimized())

        a.fitness.values = (old_mttf + (mttf - old_mttf) / N[a],
                            old_usage + (usage - old_usage) / N[a],
                            size)

    return list(zip(range(1, len(individuals) + 1), [i.fitness.values for i in individuals], list(N.values())))
