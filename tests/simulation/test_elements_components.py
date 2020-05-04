import pytest
import numpy as np

from design.application import Application
from design.component import Component
from design.designpoint import Designpoint
from simulation.elements.components import Components

"""
TODO:
- index_to_pos
"""


class TestComponents:
    @staticmethod
    def get_components_example(cap1=100, cap2=100,
           loc1=(0, 0), loc2=(1, 1),
           app1=50, app2=50):
        c1 = Component(cap1, loc1)
        c2 = Component(cap2, loc2)

        a1 = Application(app1)
        a2 = Application(app2)

        dp = Designpoint([c1, c2], [a1, a2], [(c1, a1), (c2, a2)])
        dp_data = dp.to_numpy()

        return Components(dp_data[0], dp_data[2], dp_data[3], dp_data[4])

    @staticmethod
    def slack_order_verification(policy, out):
        slack = np.asarray([[20, 0],
                            [0, 40]])

        comp_loc_map = np.asarray([(0, 0, 0), (1, 1, 1)], dtype=[('index', 'i4'), ('x', 'i4'), ('y', 'i4')])

        c = Components(slack, None, comp_loc_map, [(0, 1), (1, 1)])

        return np.array_equal(np.asarray(out), c.get_mapping_order(slack, policy))

    def test_most_slack_first_order(self):
        assert self.slack_order_verification(policy='most', out=[1, 0])

    def test_least_slack_first_order(self):
        assert self.slack_order_verification(policy='least', out=[0, 1])

    def test_alive_component(self):
        components = self.get_components_example()

        assert np.array_equal(components.alive_components, np.asarray([[True, False], [False, True]]))

    def test_com_loc_map(self):
        components = self.get_components_example()

        assert tuple(components.comp_loc_map[0]) == (0, 0, 0)
        assert tuple(components.comp_loc_map[1]) == (1, 1, 1)

    def test_step_ok(self):
        components = self.get_components_example()

        assert components.step(np.asarray([[0.5, 0.], [0., 0.5]]))

    def test_step_failure(self):
        components = self.get_components_example()

        assert not components.step(np.asarray([[1.0, 0.0], [0.0, 1.0]]))

    def test_index_to_pos(self):
        components = self.get_components_example()

        assert components.index_to_pos(0) == (0, 0)
        assert components.index_to_pos(1) == (1, 1)

    def test_get_failed_indices(self):
        components = self.get_components_example()

        failed_components = np.asarray([[True, False], [False, False]])

        assert components.get_failed_indices(failed_components).size == 1
        assert list(components.get_failed_indices(failed_components))[0] == 0

    def test_handle_failure_ok(self):
        components = self.get_components_example()

        failed_components = np.asarray([[True, False], [False, False]])

        new_mapping = components.handle_failures(failed_components)

        assert tuple(components.app_mapping[0]) == (1, 50)
        assert tuple(components.app_mapping[1]) == (1, 50)
        assert new_mapping

    def test_handle_failure_fail(self):
        components = self.get_components_example(app1=100, app2=100)

        failed_components = np.asarray([[True, False], [False, False]])

        new_mapping = components.handle_failures(failed_components)

        assert not new_mapping

