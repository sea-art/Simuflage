import time
import pytest

from design.designpoint import DesignPoint
from dse.montecarlo import monte_carlo


class TestMonteCarlo:
    def test_speed_monte_carlo_parallelized(self):
        """ Checks if a Monte Carlo simulation is fast enough.
        :return: None
        """
        dps = [DesignPoint.create_random(2) for _ in range(10)]

        start = time.time()
        monte_carlo(dps, iterations=5000, parallelized=True)
        end = time.time()

        assert end - start < 10.0

    def test_monte_carlo_iterative(self):
        """ Tests if iterative implementation of MCS works.

        :return: None
        """
        dps = [DesignPoint.create_random(2) for _ in range(3)]

        with pytest.warns(UserWarning):
            monte_carlo(dps, iterations=500, parallelized=False)
