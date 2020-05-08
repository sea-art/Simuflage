#!/usr/bin/env python

""" This file is currently mainly used for testing/debugging purposes

Use src/dse/montecarlo.py instead for the evaluation of design points.
"""
import collections
import math

from design.designpoint import Designpoint
from dse.montecarlo import monte_carlo, print_results


__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"

from simulation.simulator import Simulator

if __name__ == "__main__":
    # dps = [Designpoint.create_random(3) for _ in range(20)]
    # results = monte_carlo(dps)
    # print_results(results, dps)

    dp1 = Designpoint.create([100, 100, 100, 100], [(0, 0), (1, 0), (1, 1), (0, 1)], [10, 10, 10, 10], [(0, 0), (1, 1), (2, 2), (3, 3)])
    dp2 = Designpoint.create([100, 100, 100, 100], [(0, 0), (1, 0), (1, 1), (0, 1)], [20, 20, 20, 20], [(0, 0), (1, 1), (2, 2), (3, 3)])
    dp3 = Designpoint.create([100, 100, 100, 100], [(0, 0), (1, 0), (1, 1), (0, 1)], [30, 30, 30, 30], [(0, 0), (1, 1), (2, 2), (3, 3)])
    dp4 = Designpoint.create([100, 100, 100, 100], [(0, 0), (1, 0), (1, 1), (0, 1)], [40, 40, 40, 40], [(0, 0), (1, 1), (2, 2), (3, 3)])
    dp5 = Designpoint.create([100, 100, 100, 100], [(0, 0), (1, 0), (1, 1), (0, 1)], [50, 50, 50, 50], [(0, 0), (1, 1), (2, 2), (3, 3)])

    results = monte_carlo([dp1, dp2, dp3, dp4, dp5], iterations=5 * 50, parallelized=True)

    results = collections.OrderedDict(sorted(results.items()))

    for k, v in results.items():
        print((k+1)/10, "mttf:", v, "(years: " + str((v / (24 * 365))) + ")")


    # mttf = sum(results.values()) / len(results)
    # years = (mttf / (24 * 365))

    # print("MTTF:", mttf, " (" + str(years) + " years)")
