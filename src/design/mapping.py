#!/usr/bin/env python

""" Contains helper functions regarding various amounts of mappings that are present in
a design point, components or applications.
"""

import numpy as np
import math

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


def all_possible_pos_mappings(n):
    """ Cartesian product of all possible position values.

    :param n: amount of components
    :return: (N x 2) integer array containing all possible positions.
    """
    grid_size = int(math.ceil(math.sqrt(n)))
    x = np.arange(grid_size)

    return np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])


def verify_unique_locations(components):
    """ Verifies that all locations are unique.

    :param components: list of Component objects.
    :return: Boolean indicating that all locations of the provided components are unique.
    """
    observed_locs = [tuple(c.loc) for c in components]

    return len(observed_locs) == len(set(observed_locs))


def comp_to_loc_mapping(components):
    """ Compute a component to location mapping as a structured numpy array as
    [(comp_index, x_coord, y_coord)]

    :return: structured numpy array [(index, x, y)]
    """
    assert verify_unique_locations(components), "components do not have unique locations"

    return np.asarray([(i, components[i].loc[0], components[i].loc[1])
                       for i in range(len(components))],
                      dtype=[('index', 'i4'), ('x', 'i4'), ('y', 'i4')])


def application_mapping(components, tuple_mapping):
    """ Create a component to application mapping as a structured numpy array as
    [(component_index, application_power_required)]

    :param components: list of component objects
    :param tuple_mapping: list of tuples, mapping (Component object, Application object)
    :return: structured numpy array [(`comp`, `app`)]
    """
    app_map = np.asarray([(components.index(comp), app.power_req) for comp, app in tuple_mapping],
                         dtype=[('comp', 'i4'), ('app', 'i4')])  # dtypes are numpy indices

    return app_map
