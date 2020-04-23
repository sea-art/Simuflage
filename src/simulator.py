import numpy as np
import os


class Simulator:
    def __init__(self, designpoint):
        dp_data = designpoint.to_numpy()
        self.capacities = dp_data[0]
        self.temperatures = dp_data[1]
        self.power_uses = dp_data[2]
        self.app_mapping = dp_data[3]

        self.nr_applications = self.app_mapping.size
        self.nr_components = self.capacities.size
        self.iterations = 0

        self.failed_components = np.zeros(self.nr_components, dtype=np.bool)
        omegas = 100 * np.random.weibull(5, self.nr_components)  # repr. iterations on 100% usage when cpu will fail
        self.lambdas = 1 / omegas
        self.agings = np.zeros(self.nr_components, dtype=np.float)  # Will increment each iteration

    def iterate(self):
        """ Run one iteration within the simulator.

        :return: boolean indicating if a core has failed this iteration.
        """
        self.update_agings()
        self.fluctuate_temperatures(0.1)

        self.log_iteration("a.txt")
        self.iterations += 1
        return self.handle_failures()

    def log_iteration(self, filename_out):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        f = open(root_dir + "/../out/" + filename_out, "a+")
        f.write("%d %d %f\n" % (self.iterations, np.sum(self.capacities), np.average(self.temperatures)))
        f.close()

    def update_agings(self):
        nonzero_caps = np.nonzero(self.capacities)[0]
        workload = self.power_uses[nonzero_caps] / self.capacities[nonzero_caps]
        self.agings[nonzero_caps] += self.lambdas[nonzero_caps] * workload  # TODO should be based on temperatures instead of workload

    def fluctuate_temperatures(self, fluctuate):
        self.temperatures += np.random.uniform(-fluctuate, fluctuate, self.temperatures.size)

    def run(self, iteration_amount):
        for _ in range(iteration_amount):
            if not self.iterate():
                print("failed - TTF:", self.iterations)
                return

    def adjust_power_uses(self):
        """Updates the power_uses for components based on the application mapping (self.app_mapping)."""
        foo = np.zeros(self.nr_components)

        for a, b in self.app_mapping:
            foo[a] += b

        self.power_uses = foo

    def handle_failures(self):
        all_failed_components = self.agings >= 1.0  # Check which components have failed

        if np.any(all_failed_components[np.invert(self.failed_components)]):  # Check if failed components are already adjusted
            self.failed_components[all_failed_components] = True
            self.capacities[all_failed_components] = 0
            self.power_uses[all_failed_components] = 0

            failed_indices = np.nonzero(all_failed_components)
            # print("i:", self.iterations, "-", "cores", failed_indices[0], "have failed")

            to_map = self.app_mapping[np.isin(self.app_mapping['comp'], failed_indices[0])]  # All applications that have to be remapped
            self.app_mapping = self.app_mapping[np.isin(self.app_mapping['comp'], failed_indices[0], invert=True)]  # Removes all applications that are mapped towards failed components

            for app in to_map['app']:
                for i in np.random.permutation(np.arange(self.nr_components)[np.invert(self.failed_components)]):  # Loop over all non-failed components
                    power_available = self.capacities - self.power_uses  # Power that the components have available

                    if app < power_available[i]:
                        self.app_mapping = np.append(self.app_mapping, np.array([(i, app)],
                                                                                dtype=self.app_mapping.dtype))
                        self.adjust_power_uses()
                        break

            with np.errstate(divide='ignore', invalid='ignore'):
                grid = self.power_uses / self.capacities
            # print(grid)

        return self.app_mapping.size == self.nr_applications  # checks if all applications are mapped