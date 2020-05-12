import pytest
import random

from design import Application


class TestApplication:
    def test_correct_variables(self):
        power_req = random.randint(0, 300)

        a = Application(power_req)

        assert a.power_req == power_req

    def test_incorrect_variables(self):
        power_req = random.randint(-300, -1)

        with pytest.raises(AssertionError):
            a = Application(power_req)
