import numpy as np


class Component:
    """ Represents a CPU within the design space exploration."""

    def __init__(self, power_capacity, location, max_temp=100):
        """ Initialize a component representing a CPU.

        :param power_capacity: abstract (non-representative) value indicating the power output capacity of a component.
        :param location: (x, y) tuple of the location of the component on the grid.
                         Each component in a designpoint should have a unique location
        :param max_temp: temperature of cpu upon 100% utilization
        """
        self.max_temp = 100
        self.capacity = power_capacity
        self.loc = location
        self.electricity_usage = 0  # maybe voltages/electricity per power output?
        self.failure_rate = 0.5
        self.failed = False
