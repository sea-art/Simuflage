#!/usr/bin/env python

""" Contains helper functions regarding various amounts of mappings that are present in
a design point, components or applications.
"""
import operator
import sys

import numpy as np
import math


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


##########################
# BIN-PACKING ALGORITHMS #
##########################
class InvalidMappingError(Exception):
    def __init__(self, func_name):
        """ Error/exception to be thrown when the algorithms can not fit
        the given applications towards the given capabilities.

        :param func_name: str - function name to display within error
        """
        super().__init__("Applications can not be mapped via {} algorithm.".format(func_name))


def next_fit(capacities, applications):
    """ Creates application mapping via next_fit (NF) algorithm.

    Keeps a single component open, and maps applications in order to that component
    yntill an application does not fit, which changes the active bin to the one next
    in order.

    :param capacities: [float] - list of capabilities of components.
    :param applications: [float] - list of comp_need requirements for applications.
    :return: [(comp_idx, app_idx)] maps index of component to index of application.
    """
    bins = np.copy(capacities)

    active_bin = 0
    tup_mapping = []

    for i, app in enumerate(applications):
        while active_bin < len(bins):
            if app <= bins[active_bin]:
                tup_mapping.append((active_bin, i))
                bins[active_bin] -= app
                break

            active_bin += 1

    if len(tup_mapping) != len(applications):
        raise InvalidMappingError("next_fit")

    return tup_mapping


def first_fit(capacities, applications):
    """ Creates application mapping via first_fit (FF) algorithm.

    Each application is mapped to the first component that the application fits in.

    :param capacities: [float] - list of capabilities of components.
    :param applications: [float] - list of comp_need requirements for applications.
    :return: [(comp_idx, app_idx)] maps index of component to index of application.
    """
    bins = np.copy(capacities)
    tup_mapping = []

    for i, app in enumerate(applications):
        for j, cap in enumerate(bins):
            if app <= cap:
                tup_mapping.append((j, i))
                bins[j] -= app
                break

    if len(tup_mapping) != len(applications):
        raise InvalidMappingError("first_fit")

    return tup_mapping


def _get_highest_idx(bins, min_val, invert=False):
    """ Find the index of the bin with the highest available capacity.

    :param bins: [float] - list of float capabilities
    :param min_val: float - minimum required value for bin
    :param invert: Will change function to finding the lowest idx.
    :return: int - index
    """
    best_i = sys.maxsize

    if not invert:
        base = -1
        opp = operator.gt
    else:
        base = sys.maxsize
        opp = operator.lt

    for i, cap in enumerate(bins):
        if opp(cap, base) and cap >= min_val:
            base = cap
            best_i = i

    return best_i


def best_fit(capacities, applications, invert=False):
    """ Creates application mapping via best_fit (BF) algorithm.

    Maps applications to the component with the highest remaining capacity (that still fits).

    :param capacities: [float] - list of capabilities of components.
    :param applications: [float] - list of comp_need requirements for applications.
    :param invert: Changes best_fit algorithm to worst_fit
    :return: [(comp_idx, app_idx)] maps index of component to index of application.
    """
    bins = np.copy(capacities)
    tup_mapping = []

    for i, app in enumerate(applications):
        idx = _get_highest_idx(bins, app, invert=invert)

        try:
            if app <= bins[idx]:
                tup_mapping.append((idx, i))
                bins[idx] -= app
        except IndexError:
            break

    if len(tup_mapping) != len(applications):
        raise InvalidMappingError("best_fit")

    return tup_mapping


def worst_fit(capacities, applications):
    """ Creates application mapping via worst_fit (WF) algorithm.

    Maps applications to the component with the least remaining capacity (that still fits).

    :param capacities: [float] - list of capabilities of components.
    :param applications: [float] - list of comp_need requirements for applications.
    :return: [(comp_idx, app_idx)] maps index of component to index of application.
    """
    try:
        return best_fit(capacities, applications, invert=True)
    except InvalidMappingError:
        raise InvalidMappingError("worst_fit")
