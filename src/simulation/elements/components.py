#!/usr/bin/env python

""" Contains all logical operations that are required for the components in a simulation

The variables of components are spatially stored as a 2D numpy float arrayS.
"""

import numpy as np
import warnings

from .element import SimulatorElement

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


class Components(SimulatorElement):
    """ Contains all logical operations that are required for the components in a simulation."""
    def __init__(self, capacities, power_uses, comp_loc_map, app_mapping, policy='random'):
        """ Initializes the components for the simulator.

        :param capacities: 2D float numpy array with capacities on component positions
        :param power_uses: 2D float numpy array with power usage on component positions
        :param comp_loc_map: Structured array mapping components to locations
        :param app_mapping: Structured array mapping components to applications
        """
        self._capacities = capacities
        self._power_uses = power_uses
        self._alive_components = capacities > 0

        self._reset_values = (np.array(capacities, copy=True),
                              np.array(power_uses, copy=True),
                              np.array(app_mapping, copy=True))

        assert np.all(self._capacities >= self._power_uses), \
            "One or more components have a workload that it can not handle."

        # Mappings
        self._comp_loc_map = comp_loc_map
        self._app_mapping = app_mapping
        self._index_loc_map = {x[0]: (x[1], x[2]) for x in self._comp_loc_map}
        self._loc_index_map = {v: k for k, v in self._index_loc_map.items()}

        # Miscellaneous variables
        self._nr_applications = np.count_nonzero(self._app_mapping)
        self._nr_components = np.count_nonzero(self._capacities)

        # self._adjust_power_uses()
        self.policy = policy

    def __repr__(self):
        """ Representation of an Components object.

        :return: string - representation of this Components object
        """
        return str(np.vstack(([self._capacities.T], [self._power_uses.T])).T)

    @property
    def alive_components(self):
        """ Getter function for the alive_components instance variable.

        :return: 2D boolean array indicating working components at corresponding location.
        """
        return self. _alive_components

    @property
    def comp_loc_map(self):
        """ Getter function for the comp_loc_map instance variable.

        :return: Structured array mapping components to locations
        """
        return self._comp_loc_map

    @property
    def capacities(self):
        """ Getter function for the comp_loc_map instance variable.

        :return: 2D float numpy array with capacities on component positions
        """
        return self._capacities

    @property
    def power_uses(self):
        """ Getter function for the power_uses instance variable.

        :return: 2D float numpy array with power usage on component positions
        """
        return self._power_uses

    @property
    def app_mapping(self):
        """ Getter function for the app_mapping instance variable.

        :return: structured array [('comp'), ('app')]
        """
        return self._app_mapping

    @property
    def workload(self):
        """ Return the current workload of the components.

        :return: 2D numpy float array with values between [0.0, 1.0], indicating workload.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            workload = self.power_uses / self._capacities
            workload[np.isnan(workload)] = 0

        return workload

    def _index_to_pos(self, index):
        """ Yield tuple (y, x) of the position of the index of a component.

        :param index: integer containing the component index
        :return: tuple (y, x)
        """
        loc = self._index_loc_map[index]

        return loc[1], loc[0]

    def _pos_to_index(self, x, y):
        """ Returns the position of a component based on a given x, y coordinate.

        :param x: integer of x position
        :param y: integer of y position
        :return: integer of index of component
        """
        return self._loc_index_map[(x, y)]

    def _get_failed_indices(self, failed_components):
        """ Receive the failed indices of components.

        :param failed_components: 2D boolean array of all components that have failed.
        :return: numpy integer array containing all indices of failed components.
        """
        failed_locations = np.transpose(np.nonzero(failed_components))

        return [self._pos_to_index(loc[1], loc[0]) for loc in failed_locations]

    def _adjust_power_uses(self):
        """ Updates the power_uses for components based on the application mapping (self.app_mapping).

        :return: None
        """
        grid = np.zeros(self._capacities.shape)

        for comp, app in self._app_mapping:
            grid[self._index_to_pos(comp)] += app

        self._power_uses = grid

    def _remove_failed_components(self, failed_components):
        """ Cleanup and alter variables of components that have failed.

        :param failed_components: Numpy boolean array indicating which components have failed.
        :return: Numpy array of indices of failed components
        """
        self._alive_components[failed_components] = False
        self._capacities[failed_components] = 0
        self._power_uses[failed_components] = 0

    # def _clean_app_map(self, failed_indices):
    #     print("FAILED INDICES", failed_indices)
    #     failed_coordinates = [z for z in map(self._index_to_pos, failed_indices)]
    #
    #     for y, x in failed_coordinates:
    #         self._capacities[y, x] = 0
    #         self._power_uses[y, x] = 0

    def _adjust_app_mapping(self, failed_indices):
        """ Removes all applications that are mapped to failed components and remaps them.

        :param failed_indices: list of integers corresponding to the failed component indices.
        :return: Boolean indiciating if application could be remapped (True = OK, False = System failure).
        """
        # Removes all applications that are mapped towards failed components
        positions = np.isin(self._app_mapping['comp'], failed_indices)
        to_map = self._app_mapping[positions]
        self._app_mapping = self._app_mapping[np.invert(positions)]

        for app in to_map['app']:
            self._remap_application(app, self.policy)

        return self._app_mapping.size == self._nr_applications

    def _sort_map_slack_based(self, slack, reverse=False):
        """ Sorts the com_loc_map based on the amount of slack (highest to lowest)

        :param slack:
        :return:
        """
        lst = []

        for i, x, y in self._comp_loc_map:
            lst.append((i, slack[y, x]))

        return np.array([i[0] for i in sorted(lst, key=lambda z: z[1], reverse=reverse)])

    def _get_mapping_order(self, slack, policy):
        """ Based on a given policy, return the corresponding map order.

        :param slack: 2d numpy float array containing the slack per local component
        :param policy: string - choice of ['random', 'most', 'least] (See self.remap_application())
        :return:
        """
        if policy not in ['random', 'most', 'least']:
            warnings.warn("Policy: " + str(policy) + " is not known. Using random policy instead.")
            policy = 'random'

        if policy == 'random':
            return np.random.permutation(self._comp_loc_map['index'])
        elif policy == 'most':
            return self._sort_map_slack_based(slack, reverse=True)
        else:  # policy == 'least'
            return self._sort_map_slack_based(slack, reverse=False)

    def _remap_application(self, app, policy='random'):
        """ Remaps a given application to any of the still working components with enough slack.

        Also adjusts the power usage after application remapping.

        :param app: integer representing the power required for an application that has to be remapped.
        :param policy - choice of
            'random' - randomly-mapped   --> will map applications randomly to components
            'most'   - most-slack-first  --> will try to map applications to components with most slack first
            'least'  - least-slack-first --> will try to map applications to components with least slack first
        :return: None
        """
        components_slack = self._capacities - self._power_uses
        map_order = self._get_mapping_order(components_slack, policy)

        for i in map_order:
            y, x = self._index_to_pos(i)

            if app <= components_slack[y, x]:
                self._app_mapping = np.append(self._app_mapping, np.array([(i, app)],
                                                                          dtype=self._app_mapping.dtype))
                self._power_uses[y, x] += app
                return

    def _handle_failures(self, failed_components):
        """Look at the aging values to determine if a component has failed.
        If a component has failed, the applications that were mapped to that component will be remapped
        to components with a sufficient amount of slack based on the policy.

        :return: Boolean indicating reliability system (True = OK, False = failure)
        """
        self._remove_failed_components(failed_components)
        failed_indices = self._get_failed_indices(failed_components)

        return self._adjust_app_mapping(failed_indices)

    def step(self, cur_agings):
        """ Run one iteration regarding the component process of the simulation

        :param cur_agings: 2D numpy float array containing the current agings for components
        :return: Boolean indicating if the simulator is still up (True = OK, False = System failure).
        """
        failed_components = cur_agings > 0.9999

        if np.any(failed_components[self.alive_components]):
            return self._handle_failures(failed_components)

        return True

    def do_n_steps(self, n, cur_agings):
        """ Run n iterations regarding the component process of the simulation.

        :param n: int - amount of timesteps to take
        :param cur_agings: 2D numpy float array containing the current agings for components
        :return: Boolean indicating if the simulator is still up (True = OK, False = System failure).
        """
        return self.step(cur_agings)

    def reset(self):
        """ Resets the components back to default.

        :return: None
        """
        capacities, power_uses, app_map = self._reset_values
        self._capacities = np.array(capacities, copy=True)
        self._power_uses = np.array(power_uses, copy=True)
        self._app_mapping = np.array(app_map, copy=True)
        self._alive_components = capacities > 0
        self._adjust_power_uses()
