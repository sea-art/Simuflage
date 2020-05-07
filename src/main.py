#!/usr/bin/env python

""" This file is currently mainly used for testing/debugging purposes

Use src/dse/montecarlo.py instead for the evaluation of design points.
"""

from design.designpoint import Designpoint
from dse.montecarlo import monte_carlo, print_results

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


if __name__ == "__main__":
    dps = [Designpoint.create_random(3) for _ in range(20)]
    results = monte_carlo(dps)
    print_results(results, dps)
