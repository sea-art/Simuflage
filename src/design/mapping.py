import numpy as np


def comp_to_loc_mapping(components):
    """ Compute a component to location mapping as a structured numpy array as
    [(comp_index, x_coord, y_coord)]

    :return: structured numpy array [(index, x, y)]
    """
    return np.asarray([(i, components[i].loc[0], components[i].loc[1])
                       for i in range(len(components))],
                      dtype=[('index', 'i4'), ('x', 'i4'), ('y', 'i4')])


def application_mapping(components, app_map):
    return np.asarray([(components.index(comp), app.power_req) for comp, app in app_map],
                      dtype=[('comp', 'i4'), ('app', 'i4')])  # dtypes are numpy indices