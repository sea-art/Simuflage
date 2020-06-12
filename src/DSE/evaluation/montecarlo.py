#!/usr/bin/env python

""" Contains methods to evaluate n designpoints via a Monte Carlo simulation approach."""

import collections
import random
import math
import multiprocessing
import warnings

from simulation import Simulator

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
    consumptions = []

    sim = Simulator(designpoint)

    for i in range(iterations):
        ttf, consum = sim.run_optimized()
        TTFs.append(ttf)
        consumptions.append(consum)

    outputs[dp_name] = sum(TTFs) / len(TTFs), sum(consumptions) / len(consumptions), designpoint.evaluate_size


def monte_carlo_iterative(designpoints, iterations):
    """ Iterative implementation of the MCS to rank the given design points.

    :param designpoints: [DesignPoint object] - List of designpoint objects (the candidates).
    :param iterations: number of MC iterations to run
    :return: [float] - List of MTTF corresponding indexwise to the design points.
    """
    TTFs = {i: [] for i in range(len(designpoints))}
    consumptions = {i: [] for i in range(len(designpoints))}

    sims = [Simulator(d) for d in designpoints]

    i_per_dp = iterations // len(designpoints)

    for i in range(len(designpoints)):
        for _ in range(i_per_dp):
            ttf, consum = sims[i].run_optimized()
            TTFs[i].append(ttf)
            consumptions[i].append(consum)

    for i in range(len(TTFs)):
        TTFs[i] = sum(TTFs[i]) / len(TTFs[i])
        consumptions[i] = sum(consumptions[i]) / len(consumptions[i])

    output = {i: [] for i in range(len(designpoints))}

    for i in TTFs:
        output[i] = (TTFs[i], consumptions[i], designpoints[i].evaluate_size())

    return output


def monte_carlo_parallelized(designpoints, iterations):
    """ Parallelised implementation of the MCS.

    This function should be used over the monte_carlo_iterative due to significant decrease
    in execution time.

    :param designpoints: [DesignPoint object] - List of designpoint objects (the candidates).
    :param iterations: number of MC iterations to run
    :return: {dp_indicator: MTTF} - Dictionary of the results
    """
    i_per_dp = int(math.ceil(iterations / len(designpoints)))
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

    return return_dict


def monte_carlo(designpoints, iterations=1000, parallelized=True):
    """ Evaluation of the given design points via a Monte Carlo Simulation

    :param designpoints: [DesignPoint object] - List of designpoint objects (the candidates).
    :param iterations: number of MC iterations to run
    :param parallelized: Boolean - indicating if the results so be calculated in parallel
    :return: Dict mapping index of designpoint with corresponding evaluated values (as tuple)
    """
    if parallelized:
        return collections.OrderedDict(sorted(monte_carlo_parallelized(designpoints, iterations).items()))
    else:
        return collections.OrderedDict(monte_carlo_iterative(designpoints, iterations))


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
