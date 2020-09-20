#!/usr/bin/env python

""" Abstract representation of an application present in embedded system design points.

Applications are represented by a single comp_need requirement value, that indicates how much processing comp_need
a component will use in order to run this application.
"""


class Application:
    def __init__(self, comp_need):
        """ Initialize an application that can be mapped to a component.

        :param comp_need: The computational need required to run this application by a component.
        """
        assert comp_need >= 0, "Power requirement for applications has to be positive"

        self._comp_need = comp_need

    def __repr__(self):
        """ Representation of an Application object.

        :return: string - representation of this Application object
        """
        return "app: {}".format(self._comp_need)

    @property
    def power_req(self):
        """ Getter function for the power_req instance variable.

        :return: integer - indicating the comp_need requirement for running this application.
        """
        return self._comp_need
