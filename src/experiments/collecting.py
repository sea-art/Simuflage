import pickle
import numpy as np
import scipy.stats
import multiprocessing
import itertools

from deap import creator, base
from deap.tools import sortNondominated, selNSGA2

from DSE import monte_carlo
from DSE.evaluation import sSAR, pareto_ucb1
from DSE.evaluation.evaluation_tests import scalarized_lambda, get_all_weights
from DSE.exploration.GA import Chromosome
from DSE.exploration.GA.algorithm import initialize_sesp, GA
from experiments import AnalysisMCS

weights = (1.0, -1.0, -1.0)

creator.create("FitnessDSE_col", base.Fitness, weights=weights)
creator.create("Individual_col", Chromosome, fitness=creator.FitnessDSE_col)

S = [scalarized_lambda(w) for w in get_all_weights() if w[2] != 1.0][::2]


class CollectMCS:
    def __init__(self):
        pass

    def run_mcs(self):
        sesp = initialize_sesp()
        data = [[Chromosome.create_random(creator.Individual_mcs, sesp) for _ in range(500)] for _ in range(20)]

        pickle.dump(data, open("out/pickles/dps3.p", "wb"))
        samples = []

        for i, data_set in enumerate(data):
            print("iteration:", i)
            try:
                res = monte_carlo(data_set, iterations=1000 * len(data_set), parallelized=True, all_samples=True)
                samples.append(res)
            except:
                print("EXCEPTION")
                continue

        pickle.dump(samples, open("out/pickles/samples3.p", "wb"))


def run_ga(output, ga_i, pop_size, n_gens, nr_samples,  eval_method='mcs'):
    sesp = initialize_sesp()

    ga = GA(pop_size, n_gens, nr_samples, sesp, eval_method=eval_method)
    ga.run()

    final_pop = ga.pop
    best_candidates = sortNondominated(final_pop, 10, first_front_only=True)[0]

    output[ga_i] = {ind: np.array(ind.fitness.values) for ind in best_candidates}


class CollectGA:
    def __init__(self):
        pass

    def run_gas(self, nr_gas, pop_size, nr_gens, nr_samples):
        jobs = []

        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        for i in range(nr_gas):
            jobs.append(multiprocessing.Process(target=run_ga,
                                                args=(return_dict, i, pop_size, nr_gens, nr_samples)))

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        return return_dict


def evaluate_and_select(return_dict, identifier, dps, ref_set, nr_samples, eval_method):
    to_select = len(dps) // 2
    samples = len(dps) * nr_samples

    # Evaluation
    if eval_method == 'ssar' or eval_method == 'sSAR':
        _, ui, N = sSAR(dps, to_select, S, samples // 6)
    elif eval_method == 'pucb' or eval_method == 'pUCB':
        ui, N = pareto_ucb1(dps, to_select, samples)
    else:
        print("Running MCS:", identifier)
        ui = list(monte_carlo(dps, samples, parallelized=False).values())
        N = [nr_samples for _ in range(len(dps))]

    for individual, eval in zip(dps, ui):
        individual.fitness.values = tuple(eval)

    # Selection
    non_dom_indivs = selNSGA2(dps, to_select)
    selected_idcs = [list(dps).index(dp) for dp in non_dom_indivs]

    wrong_selected = [i for i in selected_idcs if i not in ref_set]

    print("Finished:", identifier)
    return_dict[identifier] = wrong_selected


class CollectEval:
    """Collects the data for the evaluation methods."""

    def __init__(self):
        self.analysis = AnalysisMCS()
        self.dps = list(self.analysis.data.keys())

        dps_sets = np.array_split(np.array(self.dps), 100)
        ref_sets = [self.get_references_selection(individuals) for individuals in dps_sets]
        sample_sets = list(range(10, 221, 30))

        jobs = []

        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        for j in range(1, 10):
            for samples in sample_sets:
                for idx in range(len(dps_sets[::2])):
                    print(samples, idx, j, 'mcs')
                    jobs.append(multiprocessing.Process(target=evaluate_and_select,
                                                        args=(return_dict, (samples, idx, j, 'mcs'), dps_sets[idx],
                                                              ref_sets[idx], samples, 'mcs')))

        print("jobs:", len(jobs))

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        print(return_dict)
        pickle.dump(dict(return_dict), open("out/pickles/evals.p", "wb"))

    # def get_references_selection(self, dps):
    #     means = self.analysis.means()
    #
    #     for dp in dps:
    #         dp.fitness.values = tuple(means[dp])
    #
    #     selected = list(itertools.chain(*sortNondominated(dps, len(dps)//2)))
    #     selected_idx = [list(dps).index(dp) for dp in selected]
    #
    #     return selected_idx


if __name__ == "__main__":
    CollectEval()
