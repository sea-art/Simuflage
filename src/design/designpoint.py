import numpy as np
from design.mapping import comp_to_loc_mapping, application_mapping


# TODO: mapping.py file?
class Designpoint:
    """Designpoint object representing a system to evaluate."""

    def __init__(self, comp_list, app_list, app_map):
        """ Initialize a Design point.

        :param comp_list: list of Component objects
        :param app_list: list of Application objects
        :param app_map: dictionary mapping {Component : [Appliction]}
        """
        self.components = comp_list
        self.applications = app_list
        self.application_map = app_map
        self.comp_loc_map = comp_to_loc_mapping(self.components)

    def get_grid_dimensions(self):
        """ Get the dimensions of the designpoint grid.

        :return: max dimensions (y, x) NOTE: unexpected order
        """
        return \
            np.max(self.comp_loc_map['y']) + 1, \
            np.max(self.comp_loc_map['x']) + 1

    def get_empty_grid(self):
        return np.zeros(self.get_grid_dimensions())

    def create_thermal_grid(self, capacities=None):
        if not capacities:
            capacities = [c.capacity for c in self.components]

        grid = self.get_empty_grid()

        for i, x, y in self.comp_loc_map:
            grid[y, x] = self.applications[i].power_req / self.components[i].capacity * self.components[i].max_temp

        return grid

    def create_capacity_grid(self):
        grid = self.get_empty_grid()

        for i in range(len(self.components)):
            c = self.components[i]

            assert grid[c.loc[1], c.loc[0]] == 0, "locations of components are not unique"

            grid[c.loc[1], c.loc[0]] = c.capacity

        return grid

    def to_numpy(self):
        """Return the components of a designpoint as numpy arrays.

        With this data, components (and their corresponding values) are all based on index.

        :return: (
                    numpy int array - capacities
                    numpy float array - temperatures
                    numpy integer array -  used power per component by mapped applications
                    numpy structured array - [(component_index, app_power_req]
                )
        """

        return                                       \
            self.create_capacity_grid(),             \
            self.create_thermal_grid(),              \
            self.calc_power_usage_per_component(),   \
            self.comp_loc_map,                       \
            application_mapping(self.components, self.application_map)

    def calc_power_usage_per_component(self):
        """ Calculate the power usage per compenent based on the mapped applications to the corresponding components.

        :return: numpy integer array indicating the mapped application power per component.
        """
        grid = self.get_empty_grid()

        for comp, app in self.application_map:
            grid[comp.loc[1], comp.loc[0]] += app.power_req

        return grid

