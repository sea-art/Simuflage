from experiments import AnalysisMCS
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np

from experiments.analysis import AnalysisGA


class PlottingMCS:
    def __init__(self):
        self.analysis = AnalysisMCS()

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


class PlottingGA:
    def __init__(self):
        self.analysis = AnalysisGA()
        self.analysis2 = AnalysisMCS()

    def plot_pareto_front_mttf_pe(self):
        front_values = self.analysis.pareto_front(use_objectives=[1.0, 1.0, 0.0])
        front = np.array([i.fitness.values for i in front_values])
        data = np.array(list(self.analysis.data.values()))

        mean_data = np.asarray(list(self.analysis2.means().values()))

        plt.scatter(mean_data.T[1], mean_data.T[0] / (24 * 365), marker='.', c='orange', label='random')
        plt.scatter(data.T[1], data.T[0] / (24 * 365), marker='^', c='blue', label='ga output')
        plt.scatter(front.T[1], front.T[0] / (24 * 365), marker='o', c='red', label="pareto optimal")

        plt.xlabel("energy consumption")
        plt.ylabel("MTTF in years")

        plt.legend()
        plt.show()

    def plot_pareto_front_mttf_size(self):
        front_values = self.analysis.pareto_front(use_objectives=[1.0, 0.0, 1.0])
        front = np.array([i.fitness.values for i in front_values])
        data = np.array(list(self.analysis.data.values()))

        mean_data = np.asarray(list(self.analysis2.means().values()))

        plt.scatter(mean_data.T[2], mean_data.T[0] / (24 * 365), marker='.', c='orange', label='random')
        plt.scatter(data.T[2], data.T[0] / (24 * 365), marker='^', c='blue', label='ga output')
        plt.scatter(front.T[2], front.T[0] / (24 * 365), marker='o', c='red', label="pareto optimal")

        plt.xlabel("size")
        plt.ylabel("MTTF in years")

        plt.legend()
        plt.show()

    def plot_pareto_front_pe_size(self):
        front_values = self.analysis.pareto_front(use_objectives=[0.0, 1.0, 1.0])
        front = np.array([i.fitness.values for i in front_values])
        data = np.array(list(self.analysis.data.values()))

        mean_data = np.asarray(list(self.analysis2.means().values()))

        plt.scatter(mean_data.T[1], mean_data.T[2], marker='.', c='orange', label='random')
        plt.scatter(data.T[1], data.T[2], marker='^', c='blue', label='ga output')
        plt.scatter(front.T[1], front.T[2], marker='o', c='red', label="pareto optimal")

        plt.xlabel("energy consumption")
        plt.ylabel("size")

        plt.legend()
        plt.show()


def obtain_all_plots():
    plottingMCS = PlottingMCS()
    plottingGA = PlottingGA()

    plottingMCS.objective_space_plots()
    plottingGA.plot_pareto_front_mttf_pe()
    plottingGA.plot_pareto_front_mttf_size()
    plottingGA.plot_pareto_front_pe_size()


if __name__ == "__main__":
    obtain_all_plots()
    # plotting = PlottingGA()
    # plotting.plot_pareto_front_mttf_pe()
    # plotting.plot_pareto_front_mttf_size()
    # plotting.plot_pareto_front_pe_size()
