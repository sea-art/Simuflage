import pickle
import numpy as np
import scipy.stats

from deap import creator, base

from DSE import monte_carlo
from DSE.exploration.GA import Chromosome
from DSE.exploration.GA.algorithm import initialize_sesp

weights = (1.0, -1.0, -1.0)

creator.create("FitnessDSE_mcs", base.Fitness, weights=weights)
creator.create("Individual_mcs", Chromosome, fitness=creator.FitnessDSE_mcs)


if __name__ == "__main__":
    sesp = initialize_sesp()
    data = [[Chromosome.create_random(creator.Individual_mcs, sesp) for _ in range(200)] for _ in range(5)]

    pickle.dump(data, open("out/pickles/dps3.p", "wb"))
    samples = []

    for i, data_set in enumerate(data):
        print("iteration:", i)
        try:
            res = monte_carlo(data_set, iterations=500 * len(data_set), parallelized=True, all_samples=True)
            samples.append(res)
        except:
            print("EXCEPTION")
            continue

    pickle.dump(samples, open("out/pickles/samples3.p", "wb"))
