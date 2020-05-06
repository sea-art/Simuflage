#!/usr/bin/env python

""" Abstract representation of an application present in embedded system design points.

Applications are represented by a single power requirement value, that indicates how much processing power
a component will use in order to run this application.
"""

__author__ = "Siard Keulen"
__email__ = "siardkeulen@gmail.com"
__licence__ = "GNU General Public License v3.0"


class Application:
    def __init__(self, power):
        """ Initialize an application that can be mapped to a component.

        :param power: The amount of power required to run this application by a component.
        """
        assert power >= 0, "Power requirement for applications has to be positive"

        self._power_req = power

    def __str__(self):
        """ String representation of an Application object.

        :return: string - representation of this Application
        """
        return "app: " + str(self._power_req)

    def __repr__(self):
        """ Representation of an Application object.

        :return: string - representation of this Application object
        """
        return self.__str__()

    @property
    def power_req(self):
        """ Getter function for the power_req instance variable.

        :return: integer - indicating the power requirement for running this application.
        """
        return self._power_req
