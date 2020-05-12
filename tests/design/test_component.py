import pytest
import random

from design import Component


class TestComponent:
    def test_correct_variables(self):
        power_cap = random.randint(0, 300)
        location = (random.randint(0, 10), random.randint(0, 10))
        max_temp = random.randint(1, 100)

        c = Component(power_cap, location, max_temp)

        assert c.capacity == power_cap
        assert c.loc == location
        assert c.max_temp == max_temp

    def test_invalid_variables(self):
        power_cap = random.randint(-300, -1)
        location = (random.randint(-10, -1), random.randint(-10, -1))
        max_temp = random.randint(-100, -1)

        with pytest.raises(AssertionError):
            c = Component(power_cap, location, max_temp)

