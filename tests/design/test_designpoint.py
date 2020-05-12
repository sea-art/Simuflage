import pytest
import numpy as np

from design import Application
from design import Component
from design import DesignPoint


class TestDesignpoint:
    @staticmethod
    def example_designpoint(cap1=100, cap2=100,
                            loc1=(0, 0), loc2=(1, 1),
                            app1=50, app2=50):
        c1 = Component(cap1, loc1)
        c2 = Component(cap2, loc2)

        a1 = Application(app1)
        a2 = Application(app2)

        return DesignPoint([c1, c2], [a1, a2], [(c1, a1), (c2, a2)])

    def test_grid_dimensions(self):
        dp = self.example_designpoint(loc1=(0, 2), loc2=(3, 1))

        assert dp._get_grid_dimensions() == (3, 4)

    def test_empty_grid(self):
        dp = self.example_designpoint(loc1=(0, 2), loc2=(3, 1))

        assert np.array_equal(dp._get_empty_grid(), np.zeros((3, 4)))

    def test_capacity_grid(self):
        dp = self.example_designpoint(cap1=80, cap2=120)

        correct_output = np.asarray([[80, 0],
                                     [0, 120]])

        assert np.array_equal(dp._create_capacity_grid(), correct_output)

    def test_power_usage(self):
        dp = self.example_designpoint()

        correct_power_usage = np.asarray([[50, 0],
                                         [0, 50]])

        assert np.array_equal(dp._calc_power_usage_per_component(), correct_power_usage)

    def test_to_numpy(self):
        """ Since all individual components are already tested, only tests the correct number of elements"""
        dp = self.example_designpoint()

        assert len(dp.to_numpy()) == 5
