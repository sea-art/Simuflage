from experiments import Analysis
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np


class Plotting:
    def __init__(self):
        self.analysis = Analysis()

        self.fig = plt.figure()

    def _scatter_plot(self, xss, yss, **kwargs):
        pass

    def plot_objective_space_3d(self):
        mean_data = np.asarray(list(self.analysis.means().values())).T
        fronts = self.analysis.pareto_front()
        fronts = np.array([i.fitness.values for i in fronts]).T

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(mean_data[1][::5], mean_data[2][::5], mean_data[0][::5] / (24 * 365), c='blue')
        ax.scatter(fronts[1], fronts[2], fronts[0] / (24 * 365), c='red')

        ax.set_xlabel("energy consumption")
        ax.set_ylabel("size")
        ax.set_zlabel("MTTF in years")

        plt.show()

    def objective_space_plots(self):
        mean_data = np.asarray(list(self.analysis.means().values()))
        fronts = []

        print("Obtaining all Pareto fronts")
        objectives = [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]

        for obj in objectives:
            front = (self.analysis.pareto_front(use_objectives=np.array(obj)))
            fronts.append(np.array([i.fitness.values for i in front]))

        print(len(fronts))
        fig = plt.figure()
        plt.scatter(mean_data.T[1], mean_data.T[0] / (24 * 365), c='blue', label='data point')
        plt.scatter(fronts[0].T[1], fronts[0].T[0] / (24 * 365), c='red', label='pareto optimal')
        plt.xlabel("energy consumption")
        plt.ylabel("MTTF in years")
        plt.legend()
        plt.show()
        fig.clear()

        fig = plt.figure()
        plt.scatter(mean_data.T[2], mean_data.T[0] / (24 * 365), c='blue', label='data point')
        plt.scatter(fronts[1].T[2], fronts[1].T[0] / (24 * 365), c='red', label='pareto optimal')
        plt.xlabel("size")
        plt.ylabel("MTTF in years")
        plt.legend()
        plt.show()
        fig.clear()

        fig = plt.figure()
        plt.scatter(mean_data.T[1], mean_data.T[2], c='blue', label='data point')
        plt.scatter(fronts[2].T[1], fronts[2].T[2], c='red', label='pareto optimal')
        plt.xlabel("energy consumption")
        plt.ylabel("size")
        plt.legend()
        plt.show()
        fig.clear()


if __name__ == "__main__":
    plotting = Plotting()
    plotting.plot_objective_space_3d()
