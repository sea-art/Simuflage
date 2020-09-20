
from DSE.exploration.algorithm import initialize_sesp, GA


class TestGA:
    def test_GA(self):
        """ Checks if a Monte Carlo simulation is fast enough.
        :return: None
        """
        sesp = initialize_sesp()

        ga = GA(100, 5, 1, sesp)
        ga.run()
