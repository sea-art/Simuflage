class Component:
    """ Represents a CPU within the design space exploration."""

    def __init__(self, power_capacity, location, max_temp=100):
        """ Initialize a component representing a CPU.

        :param power_capacity: abstract (non-representative) value indicating the power output capacity of a component.
        :param location: (x, y) tuple of the location of the component on the grid.
                         Each component in a designpoint should have a unique location
        :param max_temp: temperature of cpu upon 100% utilization
        """
        self._max_temp = max_temp
        self._capacity = power_capacity
        self._loc = location

    @property
    def max_temp(self):
        return self._max_temp

    @property
    def capacity(self):
        return self._capacity

    @property
    def loc(self):
        return self._loc
