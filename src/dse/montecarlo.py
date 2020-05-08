#!/usr/bin/env python

""" Contains methods to evaluate n designpoints via a Monte Carlo simulation approach."""
import collections
import random
import math
import multiprocessing
import warnings

from simulation.simulator import Simulator

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


def run_n_simulations(designpoint, dp_name, iterations, outputs):
    """ Evaluate the given design point by calculating the MTTF for a given amount of iterations.

    This function is used for parallelising the Monte Carlo Simulation evaluation.

    :param designpoint: DesignPoint object - this DesignPoint is evauluated.
    :param dp_name: any - unique inidcator of this designpoint (e.g. integer)
    :param iterations: number of iterations to run the MCS.
    :param outputs: dictionary to write the MTTF output.
    :return: None
    """
    TTFs = []
    sim = Simulator(designpoint)

    for i in range(iterations):
        TTFs.append(sim.run_optimized())

    outputs[dp_name] = sum(TTFs) / len(TTFs)


def monte_carlo_iterative(designpoints, iterations):
    """ Iterative implementation of the MCS to rank the given design points.

    :param designpoints: [DesignPoint object] - List of designpoint objects (the candidates).
    :param iterations: number of MC iterations to run
    :return: [float] - List of MTTF corresponding indexwise to the design points.
    """
    warnings.warn("Using the non-parallelized Monte Carlo evaluation. "
                  "NOTE: it is advised use monte_carlo() with parallelized=True for significant better performance.")

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
    """ Parallelised implementation of the MCS.

    This function should be used over the monte_carlo_iterative due to significant decrease
    in execution time.

    :param designpoints: [DesignPoint object] - List of designpoint objects (the candidates).
    :param iterations: number of MC iterations to run
    :return: {dp_indicator: MTTF} - Dictionary of the results
    """
    i_per_dp = int(math.ceil(iterations / len(designpoints)))
    print("Running montecarlo simulation")
    print("Total iterations:\t", iterations)
    print("Iterations per design point:\t", i_per_dp, "\n")
    jobs = []

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for i in range(len(designpoints)):
        jobs.append(multiprocessing.Process(target=run_n_simulations,
                                            args=(designpoints[i], i, i_per_dp, return_dict)))

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

    return dict(return_dict)


def monte_carlo(designpoints, iterations=10000, parallelized=True):
    """ Evaluation of the given design points via a Monte Carlo Simulation

    :param designpoints: [DesignPoint object] - List of designpoint objects (the candidates).
    :param iterations: number of MC iterations to run
    :param parallelized: Boolean - indicating if the results so be calculated in parallel
    :return: Dict of MTTF data
    """
    if parallelized:
        return collections.OrderedDict(sorted(monte_carlo_parallelized(designpoints, iterations).items()))
    else:
        return monte_carlo_iterative(designpoints, iterations)


def print_results(results, dps):
    """ Prints the parallel MCS results

    :param results: {dp_indicator: MTTF} - Dictionary of the results
    :param dps: [Desigpoint object] - list of designpoint objects (the candidates).
    :return: None
    """
    sorted_results = sorted(results, key=results.get, reverse=True)

    for k in sorted_results:
        print("DesignPoint", k, "MTTF:", results[k])

    print("\nWinner: design point", sorted_results[0], dps[sorted_results[0]])
    print("\nLoser: design point", sorted_results[-1], dps[sorted_results[-1]])
