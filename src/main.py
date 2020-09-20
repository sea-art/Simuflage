#!/usr/bin/env python

""" This file is currently mainly used for testing/debugging purposes

Use src/DSE/montecarlo.py instead for the evaluation of design points.
"""

import numpy as np

from DSE import monte_carlo
from design import DesignPoint
from design.mapping import all_possible_pos_mappings


if __name__ == "__main__":
    dps = []
    n = 2

    for i in range(1, 11):
        dps.append(DesignPoint.create(caps=np.repeat(100, n * n),
                                      locs=all_possible_pos_mappings(n * n)[:n * n],
                                      apps=np.repeat(i * 10, n * n),
                                      maps=[(i, i) for i in range(n * n)],
                                      policy='random'))

    results = monte_carlo(dps, sample_budget=len(dps) * 1000)

    for k, v in results.items():
        print("Workload: {}\t\tMTTF: {:.4f} years\t\tAvg. Power usage: {:.2f}".format((k + 1) / 10,
                                                                                      v[0] / (24 * 365),
                                                                                      v[1]))
