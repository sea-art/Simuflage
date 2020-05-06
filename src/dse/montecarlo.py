import random
import math
import numpy as np
import multiprocessing

from design.designpoint import Designpoint
from simulation.simulator import Simulator


def all_possible_pos_mappings(n):
    """ Cartesian product of all possible position values.

    :param n: amount of components
    :return: (N x 2) integer array containing all possible positions.
    """
    grid_size = int(math.ceil(math.sqrt(n)))
    x = np.arange(grid_size)

    return np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])


def run_n_simulations(designpoint, dp_name, iterations, outputs):
    TTFs = []
    sim = Simulator(designpoint)

    for i in range(iterations):
        TTFs.append(sim.run_optimized())

    outputs[dp_name] = sum(TTFs) / len(TTFs)


def monte_carlo_iterative(designpoints, iterations):
    TTFs = [[] for _ in designpoints]
    sims = [Simulator(d) for d in designpoints]

    for a in range(iterations):
        print("MC iteration:", a, end="\r")
        i = random.randint(0, len(designpoints) - 1)
        TTFs[i].append(sims[i].run_optimized())

    for i in range(len(TTFs)):
        TTFs[i] = sum(TTFs[i]) / len(TTFs[i])

    return TTFs


def monte_carlo_parallelized(designpoints, iterations):
    i_per_dp = int(math.ceil(iterations / len(designpoints)))
    print("Running montecarlo simulation")
    print("Total iterations:\t\t", iterations)
    print("Iterations per design point:\t", i_per_dp)
    jobs = []

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for i in range(len(designpoints)):
        jobs.append(multiprocessing.Process(target=run_n_simulations, args=(designpoints[i], i, i_per_dp, return_dict)))

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

    return dict(return_dict)


def monte_carlo(designpoints, iterations=10000, parallelized=True):
    if parallelized:
        return monte_carlo_parallelized(designpoints, iterations)
    else:
        return monte_carlo_iterative(designpoints, iterations)


def print_results(results, dps):
    sorted_results = sorted(results, key=results.get, reverse=True)

    for k in sorted_results:
        print("Designpoint", k, "MTTF:", results[k])

    print("Winner: design point", sorted_results[0])


def run_test():
    dps = [Designpoint.create_random(3) for _ in range(8)]

    results = monte_carlo(dps, 10000)
    print_results(results, dps)


if __name__ == "__main__":
    run_test()
