#!/usr/bin/env python

""" Abstract representation of an embedded system design point.
A design point for an adaptive embedded system consists of:
- A list of Component objects
- A list of Application objects
- A mapping indicating which Application is being executed by which Component
- An adaptive policy on the occurrence of a Component failure.
"""

import numpy as np

from design.application import Application
from design.component import Component
from design.mapping import comp_to_loc_mapping, application_mapping, all_possible_pos_mappings

__author__ = "Siard Keulen"
__email__ = "siardkeulen@gmail.com"
__licence__ = "GNU General Public License v3.0"


class Designpoint:
    """Designpoint object representing a system to evaluate."""

    def __init__(self, comp_list, app_list, app_map, policy="random"):
        """ Initialize a Design point.

        :param comp_list: list of Component objects
        :param app_list: list of Application objects
        :param app_map: List of tuples [(Component obj, Appliction obj)}
        """
        self._components = comp_list
        self._applications = app_list
        self._application_map = app_map
        self._comp_loc_map = comp_to_loc_mapping(self._components)
        self.policy = policy

    def __str__(self):
        """ String representation of an Component object.

        :return: string - representation of this Component
        """
        return "Designpoint:\n" + str(self._application_map) + "\npolicy: " + self.policy

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
        :return: Designpoint object
        """
        comps = [Component(caps[i], locs[i]) for i in range(len(caps))]
        apps = [Application(a) for a in apps]
        mapping = [(comps[maps[i][0]], apps[maps[i][1]]) for i in range(len(maps))]

        return Designpoint(comps, apps, mapping, policy=policy)

    @staticmethod
    def create_random(n):
        """ Simplified static function to quickly generate random design points.

        Creates n components with a random capacity and random location. Creates n applications with a random power
        requirement. Picks a random adaptivity policy.

        :param n: integer - representing amount of components and applications that will be randomly created.
        :return: Designpoint object
        """
        caps = np.random.randint(61, 200, n)
        locs = all_possible_pos_mappings(n)
        apps = np.random.randint(10, 60, n)
        maps = [(a, a) for a in range(n)]

        policy = np.random.choice(["random", "most", "least"])

        return Designpoint.create(caps, locs, apps, maps, policy)

    def get_grid_dimensions(self):
        """ Get the dimensions of the designpoint grid.

        :return: max dimensions (y, x) NOTE: unexpected order
        """
        return \
            np.max(self._comp_loc_map['y']) + 1, \
            np.max(self._comp_loc_map['x']) + 1

    def get_empty_grid(self):
        """ Creates a 2D numpy array (grid) of zero's based on the position of components.

        :return: 2D numpy array
        """
        return np.zeros(self.get_grid_dimensions())

    def create_thermal_grid(self):
        """ Creates a thermal grid of all the components.

        The temperatures of each component will be placed on the
        corresponding position. All other positions temperature are 0.

        :return: 2D numpy array of local temperatures
        """

        grid = self.get_empty_grid()

        for i, x, y in self._comp_loc_map:
            grid[y, x] = self._applications[i].power_req / self._components[i].capacity * self._components[i].max_temp

        return grid

    def create_capacity_grid(self):
        """ Creates a capacity grid of all the components.

        The capacity of each component will be placed on the
        corresponding position. All other position (i.e. spots where the components are not placed) are 0.

        :return: 2D numpy array of power capacities
        """
        grid = self.get_empty_grid()

        for i in range(len(self._components)):
            c = self._components[i]

            assert grid[c.loc[1], c.loc[0]] == 0, "locations of _components are not unique"

            grid[c.loc[1], c.loc[0]] = c.capacity

        return grid

    def to_numpy(self):
        """Return the components of a designpoint as numpy arrays.

        With this data, components (and their corresponding values) are all based on index.

        :return: (
                    numpy 2D float array - capacities
                    numpy 2D float array - temperatures
                    numpy 2D float array -  used power per component by mapped applications
                    numpy 2D structured array [(component_index, app_power_req] - application mapping
                    numpy 2D structured array [(component_index, x, y)] - component to pos mapping
                )
        """

        return                                       \
            self.create_capacity_grid(),             \
            self.create_thermal_grid(),              \
            self.calc_power_usage_per_component(),   \
            self._comp_loc_map,                       \
            application_mapping(self._components, self._application_map)

    def calc_power_usage_per_component(self):
        """ Calculate the power usage per component based on the mapped applications to the corresponding components.

        :return: numpy integer array indicating the mapped application power per component.
        """
        grid = self.get_empty_grid()

        for comp, app in self._application_map:
            grid[comp.loc[1], comp.loc[0]] += app.power_req

        return grid
