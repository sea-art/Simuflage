#!/usr/bin/env python

""" Abstract representation of an application present in embedded system design points.

Applications are represented by a single power requirement value, that indicates how much processing power
a component will use in order to run this application.
"""

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


class Application:
    def __init__(self, power):
        """ Initialize an application that can be mapped to a component.

        :param power: The amount of power required to run this application by a component.
        """
        assert power >= 0, "Power requirement for applications has to be positive"

        self._power_req = power

    def __repr__(self):
        """ Representation of an Application object.

        :return: string - representation of this Application object
        """
        return "app: " + str(self._power_req)

    @property
    def power_req(self):
        """ Getter function for the power_req instance variable.

        :return: integer - indicating the power requirement for running this application.
        """
        return self._power_req
