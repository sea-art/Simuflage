#!/usr/bin/env python

""" This file is currently mainly used for testing/debugging purposes

Use src/dse/montecarlo.py instead for the evaluation of design points.
"""

import numpy as np

from design.designpoint import DesignPoint
from design.mapping import all_possible_pos_mappings
from dse.montecarlo import monte_carlo

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


if __name__ == "__main__":
    print("Simulating different workloads for 2x2 homogeneous grid.\n")
    dps = []
    n = 2
    amount_dps = 10

    for i in range(1, amount_dps + 1):
        dps.append(DesignPoint.create(caps=np.repeat(100, n * n),
                                      locs=all_possible_pos_mappings(n * n),
                                      apps=np.repeat(i * (100 / amount_dps), n * n),
                                      maps=[(i, i) for i in range(n * n)],
                                      policy='random'))

    results = monte_carlo(dps, iterations=len(dps) * 1000, parallelized=True)

    for k, v in results.items():
        print("Workload:", (k + 1) / amount_dps, "\tMTTF:", np.around(v, 1), "\t(Years: " + str((v / (24 * 365))) + ")")
