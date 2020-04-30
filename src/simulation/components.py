import numpy as np


class Components:
    def __init__(self, capacities, power_uses, comp_loc_map, app_mapping):
        """ Initializes the components for the simulator.

        :param capacities: 2D float numpy array with capacities on component positions
        :param power_uses: 2D float numpy array with power usage on component positions
        :param comp_loc_map: Structured array mapping components to locations
        :param app_mapping: Structured array mapping components to applications
        """
        self.capacities = capacities
        self.power_uses = power_uses
        self.alive_components = capacities > 0

        # Mappings
        self.comp_loc_map = comp_loc_map
        self.app_mapping = app_mapping

        # Miscellaneous variables
        self.nr_applications = np.count_nonzero(self.app_mapping)
        self.nr_components = np.count_nonzero(self.capacities)

        self.adjust_power_uses()

    def iterate(self, cur_agings, iteration):
        """ Run one iteration regarding the component process of the simulation

        :param cur_agings: 2D numpy float array containing the current agings for components
        :return: Boolean indicating if the simulator is still up (True = OK, False = System failure).
        """
        handled_failure = self.handle_failures(cur_agings, iteration)

        return handled_failure

    def index_to_grid_position(self, index):
        """ Yield tuple (y, x) of the position of the index of a component.

        :param index: integer containing the component index
        :return:
        """
        pos = self.comp_loc_map['index'] == index
        loc = self.comp_loc_map[pos]

        assert loc.size == 1, "A component has multiple locations"

        loc = loc[0]  # Is an array of one element (tuple)

        return loc[2], loc[1]

    def grid_position_to_index(self, x, y):
        """ Returns the position of a component based on a given x, y coordinate.

        :param x: integer of x position
        :param y: integer of y position
        :return: integer of index of component
        """
        pos = np.logical_and(self.comp_loc_map['x'] == x, self.comp_loc_map['y'] == y)

        assert self.comp_loc_map[pos]['index'].size == 1, "No or multiple components found at given index"

        return self.comp_loc_map[pos]['index'][0]

    def adjust_power_uses(self):
        """ Updates the power_uses for components based on the application mapping (self.app_mapping).

        :return: None
        """
        grid = np.zeros(self.capacities.shape)

        for comp, app in self.app_mapping:
            grid[self.index_to_grid_position(comp)] += app

        self.power_uses = grid

    def get_failed_indices(self, failed_components):
        """ Receive the failed indices of components.

        :param failed_components: 2D boolean array of all components that have failed.
        :return: numpy integer array containing all indices of failed components.
        """
        failed_locations = np.asarray(np.nonzero(failed_components)).T
        failed_indices = np.array([self.grid_position_to_index(loc[1], loc[0]) for loc in failed_locations])

        return failed_indices

    def cleanup_failed_components(self, failed_components):
        """ Cleanup and alter variables of components that have failed.

        :param failed_components: Numpy boolean array indicating which components have failed.
        :return: Numpy array of indices of failed components
        """
        self.alive_components[failed_components] = False
        self.capacities[failed_components] = 0

    def remap_application(self, app):
        """ Remaps a given application to any of the still working components with enough slack.

        Also adjusts the power usage after application remapping.

        :param app: integer representing the power required for an application that has to be remapped.
        :return: None
        """
        components_slack = self.capacities - self.power_uses

        # Loop randomly over all non-failed components
        for i in np.random.permutation(self.comp_loc_map['index']):
            x, y = self.comp_loc_map[self.comp_loc_map['index'] == i][['x', 'y']][0]

            if app <= components_slack[y, x]:
                self.app_mapping = np.append(self.app_mapping, np.array([(i, app)],
                                                                        dtype=self.app_mapping.dtype))
                self.adjust_power_uses()  # TODO: can be speed up
                break

    def cleanup_app_mapping(self, failed_indices):
        """ Removes all applications that are mapped to failed components and remaps them.

        :param failed_indices: 2D numpy boolean array indicating which components have failed.
        :return: Boolean indiciating if application could be remapped (True = OK, False = System failure).
        """
        # Removes all applications that are mapped towards failed components
        to_map = self.app_mapping[np.isin(self.app_mapping['comp'], failed_indices)]

        self.app_mapping = self.app_mapping[np.isin(self.app_mapping['comp'], failed_indices, invert=True)]
        self.adjust_power_uses()

        for app in to_map['app']:
            self.remap_application(app)

        return self.app_mapping.size == self.nr_applications

    def handle_failures(self, cur_agings, iteration):
        """Look at the aging values to determine if a component has failed.
        If a component has failed, the applications that were mapped to that component will randomly be remapped
        to components with a sufficient amount of slack.

        :return: Boolean indicating reliability system (True = OK, False = failure)
        """
        failed_components = cur_agings >= 1.0  # All failed components

        # Check if failed components are already adjusted (i.e. any remapping required?)
        if np.any(failed_components[self.alive_components]):
            self.cleanup_failed_components(failed_components)
            failed_indices = self.get_failed_indices(failed_components)

            return self.cleanup_app_mapping(failed_indices)

        return True
