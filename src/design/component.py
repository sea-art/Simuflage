#!/usr/bin/env python

""" Abstract representation of a component present in embedded system design points.

A Component object is a simplification of a processing component (e.g. CPU) that is represented by its comp_need capacity
(which is closely related to Application objects), a location on the embedded system grid and a maximum temperature.
"""


class Component:
    """Represents a computing component within a design point."""

    def __init__(self, comp_capability, location, self_temp=50):
        """ Initialize a component representing a CPU.

        :param comp_capability: abstract (non-representative) value indicating the comp_need output capacity of a component.
        :param location: (x, y) tuple of the location of the component on the grid.
                         Each component in a designpoint should have a unique location
        :param self_temp: temperature of cpu upon 100% utilization
        """
        assert comp_capability >= 0, "Power_capacity has to be a non-negative integer"
        assert location[0] >= 0, "Location indices have to be non-negative"
        assert location[1] >= 0, "Location indices have to be non-negative"
        assert self_temp > 0, "Max_temp has to be greater than 0"

        self._self_temp = self_temp
        self._capability = comp_capability
        self._loc = location

    def __repr__(self):
        """ Representation of an Component object.

        :return: string - representation of this Component object
        """
        return "comp: {}".format(self._capability)

    @property
    def max_temp(self):
        """ Getter function for the max_temp instance variable.

        :return: integer indicating the max temperature for this component.
        """
        return self._self_temp

    @property
    def capacity(self):
        """ Getter function for the max_temp instance variable.

        :return: integer indicating the comp_need capacity for this component.
        """
        return self._capability

    @property
    def loc(self):
        """ Getter function for the max_temp instance variable.

        :return: integer tuple (x, y) indicating the position of this component.
        """
        return self._loc
