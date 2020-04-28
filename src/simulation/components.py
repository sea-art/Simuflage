import numpy as np


class Components:
    def __init__(self, capacities, power_uses, comp_loc_map, app_mapping):
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

    def iterate(self, cur_agings):
        return self.handle_failures(cur_agings)

    def index_to_grid_position(self, index):
        """ Yield tuple (y, x) of the position of the index of a component.

        :param index:
        :return:
        """
        pos = self.comp_loc_map['index'] == index
        loc = self.comp_loc_map[pos]

        assert loc.size == 1, "A component has multiple locations"

        loc = loc[0]  # Is an array of one element (tuple)

        return loc[2], loc[1]

    def grid_position_to_index(self, x, y):
        pos = np.logical_and(self.comp_loc_map['x'] == x, self.comp_loc_map['y'] == y)

        assert self.comp_loc_map[pos]['index'].size == 1, "No or multiple components found at given index"

        return self.comp_loc_map[pos]['index'][0]

    def adjust_power_uses(self):
        """ Updates the power_uses for components based on the application mapping (self.app_mapping).

        TODO: clone of designpoint.calc_power_usage_per_component
        :return:
        """
        grid = np.zeros(self.capacities.shape)

        for comp, app in self.app_mapping:
            grid[self.index_to_grid_position(comp)] += app

        self.power_uses = grid

    def clear_failed_components(self, failed_components):
        """ Cleanup and alter variables of components that have failed.

        :param failed_components: Numpy boolean array indicating which components have failed.
        :return: Numpy array of indices of failed components
        """
        # TODO: remove from comp to loc mapping?
        self.alive_components[failed_components] = False
        self.capacities[failed_components] = 0

        failed_locations = np.asarray(np.nonzero(failed_components)).T
        failed_indices = np.array([self.grid_position_to_index(loc[1], loc[0]) for loc in failed_locations])

        # Remove failed components from comp_loc_map
        # TODO: Does comp_loc_map have to be altered? Since failed components are indexed via alive_components
        if failed_indices.size > 0:
            self.comp_loc_map = np.delete(self.comp_loc_map, failed_indices)

        return failed_indices

    def handle_failures(self, cur_agings):
        """Look at the aging values to determine if a component has failed.
        If a component has failed, the applications that were mapped to that component will randomly be remapped
        to components with a sufficient amount of slack.

        :return: Boolean indicating failure occurrence.
        """
        failed_components = cur_agings >= 1.0  # All failed components

        # Check if failed components are already adjusted (i.e. any remapping required?)
        if np.any(failed_components[self.alive_components]):
            failed_indices = self.clear_failed_components(failed_components)

            # All applications that have to be remapped
            to_map = self.app_mapping[self.app_mapping['comp'] == failed_indices]

            # Removes all applications that are mapped towards failed components
            self.app_mapping = self.app_mapping[np.isin(self.app_mapping['comp'], failed_indices, invert=True)]
            self.adjust_power_uses()

            for app in to_map['app']:
                components_slack = self.capacities - self.power_uses

                # Loop randomly over all non-failed components
                for i in np.random.permutation(self.comp_loc_map['index']):
                    x, y = self.comp_loc_map[self.comp_loc_map['index'] == i][['x', 'y']][0]

                    if app <= components_slack[y, x]:
                        self.app_mapping = np.append(self.app_mapping, np.array([(i, app)],
                                                                                dtype=self.app_mapping.dtype))
                        self.adjust_power_uses()  # TODO: can be speed up
                        break

        if self.app_mapping.size != self.nr_applications:
            print("required", app)
            print("available\n", self.capacities - self.power_uses)

        return self.app_mapping.size == self.nr_applications  # checks if all applications are mapped, otherwise failure
