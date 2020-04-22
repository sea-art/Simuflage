

class Component:
    def __init__(self, power_capacity):
        self.capacity = power_capacity
        self.base_temp = 10.0
        self.electricity_usage = 0 # maybe voltages/electricity per power output?
        self.power_used = 20
        self.failure_rate = 0.5
        self.failed = False



