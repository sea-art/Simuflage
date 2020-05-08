#!/usr/bin/env python

""" Abstract representation of an embedded system design point.
A design point for an adaptive embedded system consists of:
- A list of Component objects
- A list of Application objects
- A mapping indicating which Application is being executed by which Component
- An adaptive policy on the occurrence of a Component failure.
"""

import numpy as np
import random

from design.application import Application
from design.component import Component
from design.mapping import comp_to_loc_mapping, application_mapping, all_possible_pos_mappings

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"

ENV_TEMP = 20


class DesignPoint:
    """DesignPoint object representing a system to evaluate."""

    def __init__(self, comp_list, app_list, app_map, policy="random"):
        """ Initialize a Design point.

        :param comp_list: list of Component objects
        :param app_list: list of Application objects
        :param app_map: List of tuples [(Component obj, Appliction obj)}
        """
        assert len(app_list) == len(app_map), "Not all applications are mapped."

        self._components = comp_list
        self._applications = app_list
        self._application_map = app_map
        self._comp_loc_map = comp_to_loc_mapping(self._components)
        self.policy = policy

    def __str__(self):
        """ String representation of an Component object.

        :return: string - representation of this Component
        """
        return "DesignPoint:\n" + str(self._application_map) + "\npolicy: " + self.policy

    @staticmethod
    def create(caps, locs, apps, maps, policy='random'):
        """ Simplified static function to quickly generate design points.

        No other objects (e.g. Components or Applications) have to be created in order
        to initialize this design point.

        :param caps: [integer] - List of integers representing the power capacity
        :param locs: [(integer, integer)] - List of integer tuples representing the coordinates of components
        :param apps: [integer] - List of integers representing the power requirement of applications
        :param maps: [(integer, integer)] - Indexwise mapping of component indices and application indices
        :param policy: ['random', 'most', 'least'] - adaptivity policy (see Simulator)
        :return: DesignPoint object
        """
        comp_indices = [comp for comp, _ in maps]
        app_indices = [app for _, app in maps]

        assert len(app_indices) == len(set(app_indices)), "Applications are not uniquely mapped."
        assert all(x >= 0 for x in comp_indices), "Components in the application mapping have negative indices."
        assert all(x >= 0 for x in app_indices), "Components in the application mapping have negative indices."

        comps = [Component(caps[i], locs[i]) for i in range(len(caps))]
        apps = [Application(a) for a in apps]
        mapping = [(comps[maps[i][0]], apps[maps[i][1]]) for i in range(len(maps))]

        return DesignPoint(comps, apps, mapping, policy=policy)

    @staticmethod
    def create_random(n):
        """ Simplified static function to quickly generate random design points.

        Creates n components with a random capacity and random location. Creates n applications with a random power
        requirement. Picks a random adaptivity policy.

        :param n: integer - representing amount of components and applications that will be randomly created.
        :return: DesignPoint object
        """
        choices = list(map(tuple, all_possible_pos_mappings(n)))
        caps = np.random.randint(61, 200, n)
        locs = random.sample(choices, n)
        apps = np.random.randint(10, 60, n)
        maps = [(a, a) for a in range(n)]

        policy = np.random.choice(["random", "most", "least"])

        return DesignPoint.create(caps, locs, apps, maps, policy)

    def _get_grid_dimensions(self):
        """ Get the dimensions of the designpoint grid.

        :return: max dimensions (y, x) NOTE: unexpected order
        """
        return \
            np.max(self._comp_loc_map['y']) + 1, \
            np.max(self._comp_loc_map['x']) + 1

    def _get_empty_grid(self):
        """ Creates a 2D numpy array (grid) of zero's based on the position of components.

        :return: 2D numpy array
        """
        return np.zeros(self._get_grid_dimensions())

    def _create_capacity_grid(self):
        """ Creates a capacity grid of all the components.

        The capacity of each component will be placed on the
        corresponding position. All other position (i.e. spots where the components are not placed) are 0.

        :return: 2D numpy array of power capacities
        """
        grid = self._get_empty_grid()

        for i in range(len(self._components)):
            c = self._components[i]

            assert grid[c.loc[1], c.loc[0]] == 0, "locations of _components are not unique"

            grid[c.loc[1], c.loc[0]] = c.capacity

        return grid

    def _create_self_temp_grid(self):
        """ Creates a self temp grid of all the components.

        The capacity of each component will be placed on the
        corresponding position. All other position (i.e. spots where the components are not placed) are 0.

        :return: 2D numpy array of power capacities
        """
        grid = self._get_empty_grid()

        for i in range(len(self._components)):
            c = self._components[i]

            grid[c.loc[1], c.loc[0]] = c.max_temp

        return grid

    def _calc_power_usage_per_component(self):
        """ Calculate the power usage per component based on the mapped applications to the corresponding components.

        :return: numpy integer array indicating the mapped application power per component.
        """
        grid = self._get_empty_grid()

        for comp, app in self._application_map:
            grid[comp.loc[1], comp.loc[0]] += app.power_req

        return grid

    def to_numpy(self):
        """Return the components of a designpoint as numpy arrays.

        With this data, components (and their corresponding values) are all based on index.

        :return: (
                    numpy 2D float array - capacities
                    numpy 2D float array - power_usage
                    numpy 2D float array - self_temperatures (max temperature each component can generate)
                    numpy 2D structured array [(component_index, app_power_req] - application mapping
                    numpy 2D structured array [(component_index, x, y)] - component to pos mapping
                )
        """
        capacities = self._create_capacity_grid()
        power_usage = self._calc_power_usage_per_component()
        self_temps = self._create_self_temp_grid()

        assert np.any(capacities >= power_usage), "Components have higher workload than capacity"

        return                          \
            capacities,                 \
            power_usage,                \
            self_temps,                 \
            self._comp_loc_map,         \
            application_mapping(self._components, self._application_map)
