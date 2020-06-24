#!/usr/bin/env python

""" This file is currently mainly used for testing/debugging purposes

Use src/DSE/montecarlo.py instead for the evaluation of design points.
"""

from src.DSE.exploration.GA import algorithm

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


if __name__ == "__main__":
    algorithm.main()
