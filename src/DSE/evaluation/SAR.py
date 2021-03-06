import math
import numpy as np

from simulation import Simulator

NR_OBJECTIVES = 3


def normalize(values, invert=False):
    """ Lineairly scalarizes the evaluation of a design point

    :param values: tuple of 3 elements (floats)
    :param weights: [float] - weights used for the scalarization
    :return: float - transferred multi-objective sampled values to a single value
    """
    mttf, usage, size = tuple(values)

    # Normalization by dividing via the max (of 50000 random design points)
    if not invert:
        mttf /= 720000
        usage /= 550
        size /= 36
    else:
        mttf *= 720000
        usage *= 550
        size *= 36

    return mttf, usage, size


def f_n_k(k, n, D):
    log_d = 1 / 2 + sum([1 / i for i in range(2, D + 1)])

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
        samples = int(f_n_k(k, nr_samples, D) - f_n_k(k - 1, nr_samples, D))
        for i in A:
            for _ in range(samples):
                new_sample = l_scale(list(simulators[i].run()) + [individuals[i].evaluate_size()],
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
    LOG_K = 1 / 2 + sum([1 / i for i in range(2, K + 1)])

    for k in range(1, K):
        n_k_prev = n_k
        n_k = math.ceil(1 / LOG_K * (n - K) / (K + 1 - k))

        samples = int(n_k - n_k_prev)  # Number of samples per phase

        for i in A:  # for all active bandits
            for _ in range(samples):  # Sample each bandit and update empirical vector
                N[i] += 1
                ui[i] = update_empirical_mean(sims[i].run(), ui[i], N[i])

        for i in range(len(S)):  # for each scalarization function dismiss a bandit
            A_j = A_all[i]
            scalarized_rewards = [(z, S[i](ui[z])) for z in A_j if z in A_j]

            idx = min(scalarized_rewards, key=lambda t: t[1])[0]
            A_j.remove(idx)  # deletes the worst individual of this scalarization func

        A = set().union(*A_all)  # Updates A to only include any arm that has not yet been removed by an F_j

    print("Accepted:", A)
    return list(zip(ui, N))


def linear_scalarize(vector, weights):
    return sum([vector[i] * weights[i] for i in range(len(vector))])


def delta_pk_ij(ui, A, f_p, p):
    """

    @param ui: empricical means (list of reward vectors)
    @param A: Non rejected arms
    @param f_p: scalarization function
    @param p: how many arms to still accept
    @return:
    """
    t_ui = [(i, f_p(ui[i])) for i in range(len(ui))]
    sorted_indices = [i for (i, val) in sorted(t_ui, key=lambda x: x[1], reverse=True) if i in A]

    i_star_up = sorted_indices[p]
    i_star_down = sorted_indices[p + 1]
    gaps = [0 for _ in range(len(t_ui))]

    for i in sorted_indices[:p]:
        gaps[i] = t_ui[i][1] - t_ui[i_star_down][1]

    for i in sorted_indices[p:]:
        gaps[i] = t_ui[i_star_up][1] - t_ui[i][1]

    assert np.any(np.array(gaps) > 0.0), (p, sorted_indices, [t_ui[z] for z in sorted_indices], f_p([1, 3, 9]))

    # return the index of the maximum gap and if this arm should be accepted
    j = gaps.index(max(gaps))

    return j, j == sorted_indices[0]


def sSAR(individuals, p, S, n):
    """ Scalarized SAR implementation as presented in algorithm 3 of [Drugan&Nowe2014]

    :param individuals: list of design points
    :param p: number of individuals to select
    :param S: list of scalarized functions
    :param n: total number of samples
    :return: Set of selected candidates
    """
    accepted_arms = [set() for _ in range(len(S))]

    K = len(individuals)  # Number of rounds
    A_all = [[i for i in range(K)] for _ in range(len(S))]  # Contains bandits for each scalarization function
    A = [i for i in range(K)]  # Total active arms
    P_i = [p for _ in range(len(S))]
    sims = [Simulator(i) for i in individuals]  # Simulators per individual

    N = [0 for _ in range(K)]  # Number of samples per individual
    ui = [[0 for _ in range(NR_OBJECTIVES)] for _ in range(K)]  # empirical reward vector

    n_k = 0
    LOG_K = 1 / 2 + sum([1 / i for i in range(2, K + 1)])

    for k in range(1, K):
        n_k_prev = n_k
        n_k = math.ceil((1 / LOG_K) * ((n - K) / (K + 1 - k)))
        samples = int(n_k - n_k_prev)  # Number of samples per individual in this phase

        for i in A:  # for all active bandits
            for _ in range(samples):  # Sample each bandit and update empirical vector
                N[i] += 1
                reward_vector = sims[i].run()
                ui[i] = update_empirical_mean(normalize(reward_vector), ui[i], N[i])

        for i in range(len(S)):
            max_gap_idx, accepted = delta_pk_ij(ui, A_all[i], S[i], P_i[i] - len(accepted_arms[i]))
            A_all[i].remove(max_gap_idx)

            if accepted:  # Store the arms that are accepted by a function for this round
                accepted_arms[i].add(max_gap_idx)

        # Updates A to only include any arm that has not yet been removed by an F_j
        A = set().union(*A_all)

    # invert normalization of mean values
    ui = [normalize(i, invert=True) for i in ui]

    return set.union(*accepted_arms), ui, N
