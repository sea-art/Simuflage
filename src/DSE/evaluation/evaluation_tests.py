import itertools
import math
import time

from deap.tools import sortNondominated, selNSGA2

from DSE.evaluation import monte_carlo, sSAR, pareto_ucb1
from DSE.evaluation.SAR import linear_scalarize

# Number of MCS to determine the mean value to be used as reference

NR_SAMPLES_REFERENCE = 300


def get_all_weights(steps=10):
    """ Will create a list of all possible weight values for the scalarization functions.

    :type steps: int representing the stepsize (as 1 / steps)
    :return:
    """
    boxes = 3
    values = steps

    rng = list(range(values + 1)) * boxes
    ans = sorted(set(i for i in itertools.permutations(rng, boxes) if sum(i) == values))

    return list(map(lambda tup: (tup[0]/steps, tup[1]/steps, tup[2]/steps), ans))


def get_best_arms(individuals, m, samples_per_dp):
    """ Get the 'correct' reference best m arms

    :param individuals: list of design points
    :param m: amount of individuals to select
    :param samples_per_dp: integer representing the total amount of samples
    :return:
    """
    ui = monte_carlo(individuals, len(individuals) * samples_per_dp, parallelized=True)

    for i in ui:
        ga.pop[i].fitness.values = ui[i]

    # Best arms are determined via NSGA2
    best_arms = list(itertools.chain(*sortNondominated(individuals, m)))
    top_m_indices = sorted([individuals.index(indiv) for indiv in best_arms])

    return ui.values(), top_m_indices


def scalarized_lambda(w):
    return lambda vec: linear_scalarize(vec, w)


def get_output_algorithms(individuals, m, nr_samples):
    """ Runs the three algorithms and return the results

    :param individuals: list of design points
    :param m: amount of individuals to select
    :param nr_samples: integer representing the total amount of samples
    :return: tuple of 7 elements
    """
    # Scalarization functions for SAR
    S = [scalarized_lambda(w) for w in get_all_weights() if w != (0, 0, -1)]

    print("|S|:", len(S))

    # Running all the evaluation algorithms
    sar_selection, sar_ui, sar_N = sSAR(individuals, m, S, nr_samples)
    actual_nr_sar_samples = sum(sar_N)  # SAR has a dynamic sample usage, it is not consistent

    print("Using:", actual_nr_sar_samples, "samples")
    ucb_ui, ucb_N = pareto_ucb1(individuals, m, actual_nr_sar_samples)

    mcs_results = monte_carlo(individuals, actual_nr_sar_samples, parallelized=True)
    mcs_ui, mcs_N = list(mcs_results.values()), \
                    [int(math.ceil(nr_samples / len(individuals))) for _ in range(len(individuals))]

    return ucb_ui, ucb_N, sar_selection, sar_ui, sar_N, mcs_ui, mcs_N


def set_fitness_values(individuals, ui):
    for i in range(len(individuals)):
        individuals[i].fitness.values = ui[i]


# ~ TEST 1 ~
def percentage_samples_best_arms(individuals, m, nr_samples):
    """ For each evaluation method, determines what percentage of samples are spent towards
    the top m arms.

    Takes a large initial population and will run a MCS with a lot of samples to more closely determine
    the actual top m arms. The empirical mean of these samples will be used as the reference for the actual
    ranking of the arms.

    Will run all the evaluation algorithms while observing what gets sampled. The output is the percentage
    of samples spent towards the top m arms.

    :param individuals: random population of designpoints
    :param m: amount of best arms
    :return:
    """
    # Get the indices of the top m arms (and their respective means).
    actual_means, top_m_indices = get_best_arms(individuals, m, NR_SAMPLES_REFERENCE)

    ucb_ui, ucb_N, _, sar_ui, sar_N, mcs_ui, mcs_N = get_output_algorithms(individuals, m, nr_samples)

    ucb_correct_samples = sum([ucb_N[i] for i in top_m_indices])
    sar_correct_samples = sum([sar_N[i] for i in top_m_indices])
    mcs_correct_samples = sum([mcs_N[i] for i in top_m_indices])

    print("~~Percentage of samples spent towards the top m optimal arms~~")
    print("UCB1:", ucb_correct_samples / nr_samples * 100, "%")
    print("sSAR:", sar_correct_samples / nr_samples * 100, "%")
    print("MCS:", mcs_correct_samples / nr_samples * 100, "%")

    return ucb_correct_samples, sar_correct_samples, mcs_correct_samples


def time_function(f, *args):
    """ Times the execution time of a given function.

    :param f: function
    :param args: arguments for function f
    :return: time of execution
    """
    start_time = time.time()
    f(*args)
    end_time = time.time()

    return end_time - start_time


# ~ Test 2 ~
def execution_times_algorithms(individuals, m, nr_samples, repeat=1):
    """ Small test to time the execution times of each method.

    :param individuals: list of design points
    :param m: amount of individuals to select
    :param nr_samples: integer representing the total amount of samples
    :param repeat: integer representing the amount of times the functions will
                   execute (takes the average of all execution times)
    :return:
    """
    evaluation_functions = [pareto_ucb1, lambda x, y, z:  sSAR(x, y, S, z), lambda x, _, z: monte_carlo(x, z, parallelized=False)]
    exec_times = [time_function(f, individuals, m, nr_samples) for f in evaluation_functions]
    func_names = ["pUCB1", "sSAR", "MCS"]

    for s in range(2, repeat):
        print("Iteration execution timing:", s)
        for i in range(len(evaluation_functions)):
            t = time_function(evaluation_functions[i], individuals, m, nr_samples)
            exec_times[i] += (t - exec_times[i]) / s

    for i in range(len(func_names)):
        print("{}: {:.3f}".format(func_names[i], exec_times[i] * 1000))


# ~ Test 3 ~
def accuracy_selected_individuals(individuals, m, nr_samples):
    """ Uses the reference points (via MCS with many samples) to create the set of the actual best arms.
    The selected arms of each of the individual functions (UCB1, sSAR, MCS) will be compared to
    this reference set.

    :param individuals: list of design points
    :param m: amount of individuals to select
    :param nr_samples: integer representing the total amount of samples
    :return: accuracy in percentage of selected individuals
    """
    actual_means, top_m_indices = get_best_arms(individuals, m, NR_SAMPLES_REFERENCE)
    ucb_ui, ucb_N, sar_selection, sar_ui, sar_N, mcs_ui, mcs_N = get_output_algorithms(individuals, m, nr_samples)

    determined_best_arms = []

    for results in [ucb_ui, mcs_ui, sar_ui]:
        set_fitness_values(individuals, results)
        determined_best_arms.append(sorted([individuals.index(i) for i in selNSGA2(individuals, m)]))

    determined_best_arms.append(list(sar_selection))
    correct_selected = [[selected[i] for i in range(m) if selected[i] in top_m_indices]
                        for selected in determined_best_arms]

    for i in range(len(actual_means)):
        individuals[i].fitness.values = list(actual_means)[i]

    z = sortNondominated(individuals, len(individuals))
    rank_index = [0 for _ in range(len(individuals))]

    for i in range(len(z)):
        for j in range(len(z[i])):
            rank_index[individuals.index(z[i][j])] = i

    print("ucb", len(correct_selected[0]), len(correct_selected[0]) / m, "% correct")
    print("mcs", len(correct_selected[1]), len(correct_selected[1]) / m, "% correct")
    print("sar NSGA2", len(correct_selected[2]), len(correct_selected[2]) / m, "% correct")

    # # Will print each individual of the population with several information
    # # (e.g. if it was selected by sSAR or MCS etc)
    # for i in range(len(individuals)):
    #     print("\nInvidiual:", i)
    #     print("Rank:", rank_index[i])
    #     print("------------------------------------------------")
    #     print(" MCS:", list(actual_means)[i])
    #     print("sSAR:", tuple(list(normalize(sar_ui[i], invert=True))))
    #     print("   N:", sar_N[i])
    #     print("------------------------------------------------")
    #     print(" MCS sel:", i in top_m_indices, "\n")
    #     print("sSAR sel:", i in sar_selection)


