import numpy as np


class Simulator:
    def __init__(self, designpoint):
        numpy_data = designpoint.to_numpy()
        self.capacities = numpy_data[0]
        self.temperatures = numpy_data[1]
        self.power_uses = numpy_data[2]
        self.failed_components = np.zeros(numpy_data[0].size, dtype=bool)
        self.iterations = 0

        self.failure_times = 100 * np.random.weibull(5, numpy_data[0].size)

    def iterate(self):
        self.fluctuate_temperatures(0.1)
        self.iterations += 1
        self.log_iteration("a.txt")
        return self.handle_failures()

    def log_iteration(self, filename_out):
        f = open("a.txt", "a+")
        f.write("%d %d %f\n" % (self.iterations, np.sum(self.capacities), np.average(self.temperatures)))
        f.close()

    def fluctuate_temperatures(self, fluctuate):
        self.temperatures += np.random.uniform(-fluctuate, fluctuate, self.temperatures.size)

    def run(self, iteration_amount):
        for _ in range(iteration_amount):
            if not self.iterate():
                print("failed - TTF:", self.iterations)
                return

    def handle_failures(self):
        if np.any(self.failure_times < self.iterations):
            print("i", self.iterations, np.sum(self.failure_times < self.iterations), "core(s) failed, remapping!")
            handling_components = self.failure_times < self.iterations
            self.failed_components[handling_components] = True
            self.capacities[handling_components] = 0
            remap_power = self.power_uses[handling_components]
            self.power_uses[handling_components] = 0
            self.failure_times[handling_components] = np.inf

            for power in remap_power:
                for i in range (self.capacities.size):
                    if not self.failed_components[i]:
                        power_available = self.capacities - self.power_uses
                        if power < power_available[i]:
                            self.power_uses[i] += power
                            return True
                return False # could not remap failures
        return True