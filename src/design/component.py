#!/usr/bin/env python

""" Abstract representation of a component present in embedded system design points.

A Component object is a simplification of a processing component (e.g. CPU) that is represented by its power capacity
(which is closely related to Application objects), a location on the embedded system grid and a maximum temperature.
"""

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


class Component:
    """Represents a computing component within a design point."""

    def __init__(self, power_capacity, location, max_temp=100):
        """ Initialize a component representing a CPU.

        :param power_capacity: abstract (non-representative) value indicating the power output capacity of a component.
        :param location: (x, y) tuple of the location of the component on the grid.
                         Each component in a designpoint should have a unique location
        :param max_temp: temperature of cpu upon 100% utilization
        """
        assert power_capacity >= 0, "Power_capacity has to be a non-negative integer"
        assert location[0] >= 0, "Location indices have to be non-negative"
        assert location[1] >= 0, "Location indices have to be non-negative"
        assert max_temp > 0, "Max_temp has to be greater than 0"

        self._max_temp = max_temp
        self._capacity = power_capacity
        self._loc = location

    def __str__(self):
        """ String representation of an Component object.

        :return: string - representation of this Component
        """
        return "comp: " + str(self._capacity) + "->" + str(self._loc)

    def __repr__(self):
        """ Representation of an Component object.

        :return: string - representation of this Component object
        """
        return "\n" + self.__str__()

    @property
    def max_temp(self):
        """ Getter function for the max_temp instance variable.

        :return: integer indicating the max temperature for this component.
        """
        return self._max_temp

    @property
    def capacity(self):
        """ Getter function for the max_temp instance variable.

        :return: integer indicating the power capacity for this component.
        """
        return self._capacity

    @property
    def loc(self):
        """ Getter function for the max_temp instance variable.

        :return: integer tuple (x, y) indicating the position of this component.
        """
        return self._loc
