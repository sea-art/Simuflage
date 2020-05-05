import numpy as np
from design.mapping import comp_to_loc_mapping, application_mapping


class Designpoint:
    """Designpoint object representing a system to evaluate."""

    def __init__(self, comp_list, app_list, app_map, policy="random"):
        """ Initialize a Design point.

        :param comp_list: list of Component objects
        :param app_list: list of Application objects
        :param app_map: dictionary mapping {Component : [Appliction]}
        """
        self._components = comp_list
        self._applications = app_list
        self._application_map = app_map
        self._comp_loc_map = comp_to_loc_mapping(self._components)
        self.policy = policy

    def __str__(self):
        return "policy: " + self.policy + "\n" + str(self._application_map)

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

