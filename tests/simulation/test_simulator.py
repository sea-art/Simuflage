import pytest

from design import Application
from design import Component
from design import DesignPoint
from simulation import Simulator


class TestSimulator:
    @staticmethod
    def get_simulator_example(cap1=100, cap2=100,
                               loc1=(0, 0), loc2=(1, 1),
                               app1=50, app2=50):
        c1 = Component(cap1, loc1)
        c2 = Component(cap2, loc2)

        a1 = Application(app1)
        a2 = Application(app2)

        dp = DesignPoint([c1, c2], [a1, a2], [(c1, a1), (c2, a2)])
        return Simulator(dp)

    def test_timestep(self):
        sim = self.get_simulator_example()

        sim.step()

        assert sim.timesteps == 1
