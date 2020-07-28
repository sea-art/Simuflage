import math

from design import DesignPoint
from simulation import Simulator

NR_OBJECTIVES = 2


def l_scale(values, weights=(1, 1)):
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


def f_n_k(k, n, D, log_d):
    if k == 0:
        return 0

    return math.ceil((1 / log_d) * ((n - D) / (D + 1 - k)))


def SAR(individuals, m, nr_samples=1000):
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
        samples = int(f_n_k(k, nr_samples, D, LOG_D) - f_n_k(k - 1, nr_samples, D, LOG_D))
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


def update_empirical_mean(vector, ui, N):
    """

    @param vector: new reward vector
    @param ui: current empirical mean
    @param N: number of samples
    @return: new empirical mean vector
    """
    for i in range(len(vector)):
        ui[i] = ui[i] + (vector[i] - ui[i]) / N

    return ui


def esSR(individuals, S, n):
    """ Efficient Scalarized Succesive Reject multi-armed bandit algorithm.
    Algorithm2 as presented in [Drugan & Nowe (2014)].

    Samples a list of individual design points in phases and will reject a design point
    per phase with multiple scalarization functions (S).
    At the end only one design point will remain, which is considered the best arm.

    :param individuals: list of design points
    :param S: list of scalarized functions
    :param n: total number of samples
    :return:
    """
    K = len(individuals)
    A = [i for i in range(K)]  # Contains all active bandits
    A_all = [[i for i in range(K)] for _ in range(len(S))]  # Contains bandits for each scalarization function
    N = [0 for _ in range(K)]  # Number of samples per bandit
    ui = [[0 for _ in range(NR_OBJECTIVES)] for _ in range(K)]  # empirical reward vector
    sims = [Simulator(i) for i in individuals]

    n_k = 0
    LOG_K = 1/2 + sum([1 / i for i in range(2, K + 1)])

    for k in range(1, K):
        n_k_prev = n_k
        n_k = math.ceil(1 / LOG_K * (n - K) / (K + 1 - k))
        samples = int(n_k - n_k_prev)  # Number of samples per phase

        for i in A:  # for all active bandits
            for _ in range(samples):  # Sample each bandit and update empirical vector
                N[i] += 1
                ui[i] = update_empirical_mean(sims[i].run_optimized(), ui[i], N[i])

        for i in range(len(S)):  # for each scalarization function dismiss a bandit
            A_j = A_all[i]
            scalarized_rewards = [(z, S[i](ui[z])) for z in A_j if z in A_j]

            idx = min(scalarized_rewards, key=lambda t: t[1])[0]
            A_j.remove(idx)  # deletes the worst individual of this scalarization func

        A = set().union(*A_all)  # Updates A to only include any arm that has not yet been removed by an F_j

    print("Accepted:", A)
    return list(zip(ui, N))


def sSAR(individuals, p, S, n):
    """

    :param individuals: list of design points
    :param p: number of individuals to select
    :param S: list of scalarized functions
    :param n: total number of samples
    :return:
    """
    K = len(individuals)
    sims = [Simulator(i) for i in individuals]
    A_all = [[i for i in range(K)] for _ in range(len(S))]  # Contains bandits for each scalarization function
    A = [i for i in range(K)]  # Contains all active bandits
    N = [0 for _ in range(K)]  # Number of samples per bandit
    ui = [[0 for _ in range(NR_OBJECTIVES)] for _ in range(K)]  # empirical reward vector


def linear_scalarize(vector, weights):
    return sum([vector[i] * weights[i] for i in range(len(vector))])


if __name__ == "__main__":
    individuals = [DesignPoint.create_random(2) for _ in range(5)]

    S = [lambda vec: linear_scalarize(vec, weights=(0.5, 0.5)),
         lambda vec: linear_scalarize(vec, weights=(0.25, 0.75)),
         lambda vec: linear_scalarize(vec, weights=(0.75, 0.25))]

    ui = esSR(individuals, S, 1000)

    print(ui)
