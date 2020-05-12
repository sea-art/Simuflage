import pytest
import numpy as np
import random

from design import Application
from design import Component
from design.mapping import comp_to_loc_mapping, application_mapping


class TestMapping:
    def test_corresponding_location(self):
        c1 = Component(100, (0, 0))
        c2 = Component(100, (0, 1))

        comp_to_loc = comp_to_loc_mapping([c1, c2])

        assert tuple(comp_to_loc[0]) == (0, 0, 0)
        assert tuple(comp_to_loc[1]) == (1, 0, 1)

    def test_unique_locations(self):
        c1 = Component(random.randint(1, 200), (random.randint(0, 5), random.randint(0, 5)))
        c2 = Component(random.randint(1, 200), (random.randint(6, 11), random.randint(6, 11)))

        comp_to_loc_mapping([c1, c2])

    def test_duplicate_locations(self):
        c1 = Component(random.randint(1, 200), (1, 2))
        c2 = Component(random.randint(1, 200), (1, 2))

        with pytest.raises(AssertionError):
            comp_to_loc_mapping([c1, c2])

    def test_application_mapping(self):
        c1 = Component(100, (0, 1))
        c2 = Component(100, (1, 0))
        a1 = Application(60)
        a2 = Application(70)

        input_mapping = [(c1, a1), (c2, a2)]
        app_mapping = application_mapping([c1, c2], input_mapping)

        assert np.array_equal(app_mapping['comp'], np.array([0, 1]))
        assert np.array_equal(app_mapping['app'], np.array([60, 70]))
