import numpy as np
from simulation.elements.element import SimulatorElement


class Components(SimulatorElement):
    def __init__(self, capacities, power_uses, comp_loc_map, app_mapping):
        """ Initializes the components for the simulator.

        :param capacities: 2D float numpy array with capacities on component positions
        :param power_uses: 2D float numpy array with power usage on component positions
        :param comp_loc_map: Structured array mapping components to locations
        :param app_mapping: Structured array mapping components to applications
        """
        self._capacities = capacities
        self._power_uses = power_uses
        self._alive_components = capacities > 0

        # Mappings
        self._comp_loc_map = comp_loc_map
        self._app_mapping = app_mapping

        # Miscellaneous variables
        self._nr_applications = np.count_nonzero(self._app_mapping)
        self._nr_components = np.count_nonzero(self._capacities)

        self.adjust_power_uses()

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

    def step(self, cur_agings):
        """ Run one iteration regarding the component process of the simulation

        :param cur_agings: 2D numpy float array containing the current agings for components
        :return: Boolean indicating if the simulator is still up (True = OK, False = System failure).
        """
        failed_components = cur_agings >= 1.0

        if np.any(failed_components[self.alive_components]):
            return self.handle_failures(failed_components)

        return True

    def index_to_pos(self, index):
        """ Yield tuple (y, x) of the position of the index of a component.

        :param index: integer containing the component index
        :return: tuple (y, x)
        """
        pos = self._comp_loc_map['index'] == index
        loc = self._comp_loc_map[pos]

        assert loc.size == 1, "A component has multiple locations"

        loc = loc[0]  # Is an array of one element (tuple)

        return loc[2], loc[1]

    def pos_to_index(self, x, y):
        """ Returns the position of a component based on a given x, y coordinate.

        :param x: integer of x position
        :param y: integer of y position
        :return: integer of index of component
        """
        pos = np.logical_and(self._comp_loc_map['x'] == x, self._comp_loc_map['y'] == y)

        assert self._comp_loc_map[pos]['index'].size == 1, "No or multiple components found at given index"

        return self._comp_loc_map[pos]['index'][0]

    def adjust_power_uses(self):
        """ Updates the power_uses for components based on the application mapping (self.app_mapping).

        :return: None
        """
        grid = np.zeros(self._capacities.shape)

        for comp, app in self._app_mapping:
            grid[self.index_to_pos(comp)] += app

        self._power_uses = grid

    def get_failed_indices(self, failed_components):
        """ Receive the failed indices of components.

        :param failed_components: 2D boolean array of all components that have failed.
        :return: numpy integer array containing all indices of failed components.
        """
        failed_locations = np.asarray(np.nonzero(failed_components)).T
        failed_indices = np.array([self.pos_to_index(loc[1], loc[0]) for loc in failed_locations])

        return failed_indices

    def remove_failed_components(self, failed_components):
        """ Cleanup and alter variables of components that have failed.

        :param failed_components: Numpy boolean array indicating which components have failed.
        :return: Numpy array of indices of failed components
        """
        self._alive_components[failed_components] = False
        self._capacities[failed_components] = 0

    def adjust_app_mapping(self, failed_indices):
        """ Removes all applications that are mapped to failed components and remaps them.

        :param failed_indices: 2D numpy boolean array indicating which components have failed.
        :return: Boolean indiciating if application could be remapped (True = OK, False = System failure).
        """
        # Removes all applications that are mapped towards failed components
        to_map = self._app_mapping[np.isin(self._app_mapping['comp'], failed_indices)]

        self._app_mapping = self._app_mapping[np.isin(self._app_mapping['comp'], failed_indices, invert=True)]
        self.adjust_power_uses()

        for app in to_map['app']:
            self.remap_application(app)

        return self._app_mapping.size == self._nr_applications

    def remap_application(self, app):
        """ Remaps a given application to any of the still working components with enough slack.

        Also adjusts the power usage after application remapping.

        :param app: integer representing the power required for an application that has to be remapped.
        :return: None
        """
        components_slack = self._capacities - self._power_uses

        # Loop randomly over all non-failed components
        for i in np.random.permutation(self._comp_loc_map['index']):
            x, y = self._comp_loc_map[self._comp_loc_map['index'] == i][['x', 'y']][0]

            if app <= components_slack[y, x]:
                self._app_mapping = np.append(self._app_mapping, np.array([(i, app)],
                                                                        dtype=self._app_mapping.dtype))
                self.adjust_power_uses()  # TODO: can be speed up
                break

    def handle_failures(self, failed_components):
        """Look at the aging values to determine if a component has failed.
        If a component has failed, the applications that were mapped to that component will randomly be remapped
        to components with a sufficient amount of slack.

        :return: Boolean indicating reliability system (True = OK, False = failure)
        """
        # Check if failed components are already adjusted (i.e. any remapping required?)
        self.remove_failed_components(failed_components)
        failed_indices = self.get_failed_indices(failed_components)

        return self.adjust_app_mapping(failed_indices)
