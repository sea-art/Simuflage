import pickle
import numpy as np
import scipy.stats
import multiprocessing


from deap import creator, base
from deap.tools import sortNondominated

from DSE import monte_carlo
from DSE.exploration.GA import Chromosome
from DSE.exploration.GA.algorithm import initialize_sesp, GA

weights = (1.0, -1.0, -1.0)

creator.create("FitnessDSE_col", base.Fitness, weights=weights)
creator.create("Individual_col", Chromosome, fitness=creator.FitnessDSE_col)


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


if __name__ == "__main__":
    runner = CollectGA()

    results = runner.run_gas(5, 100, 5, 500)

    print(type(results))
    p_results = dict(results)

    print(type(p_results))

    pickle.dump(p_results, open("out/pickles/refga.p", "wb"))
