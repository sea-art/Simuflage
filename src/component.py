class Component:
    """ Represents a CPU within the design space exploration."""

    def __init__(self, power_capacity):
        """ Initialize a component representing a CPU.

        :param power_capacity: abstract (non-representative) value indicating the power output capacity of a component.
        """
        self.capacity = power_capacity
        self.base_temp = 10.0
        self.electricity_usage = 0  # maybe voltages/electricity per power output?
        self.failure_rate = 0.5
        self.failed = False
