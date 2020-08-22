import pickle
import numpy as np
import scipy.stats
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from deap import creator, base
from deap.tools import sortNondominated

from DSE.exploration.GA import Chromosome



weights = (1.0, -1.0, -1.0)

creator.create("FitnessDSE_mcs", base.Fitness, weights=weights)
creator.create("Individual_mcs", Chromosome, fitness=creator.FitnessDSE_mcs)


class Analysis:
    def __init__(self):
        print("Reading data")
        dps = pickle.load(open("out/pickles/dps3.p", "rb"))
        samples = pickle.load(open("out/pickles/samples3.p", "rb"))

        print("Adjusting data")
        self.data = {}

        self.mean_data = None
        self.confidence_interval_data = None
        self.pareto_front_data = None

        for i, s in enumerate(samples):
            for k, v in s.items():
                self.data[dps[i][k]] = np.asarray(v)

    def means(self):
        if self.mean_data is not None:
            return self.mean_data

        mean_data = {}

        for k, v in self.data.items():
            mean_data[k] = np.mean(v, axis=0)

        self.mean_data = mean_data

        return mean_data

    def confidence_intervals(self, confidence=0.95):
        if self.confidence_interval_data is not None:
            return self.confidence_interval_data

        conf_data = {}

        for k, v in self.data.items():
            mttf = 1.0 * v.T[0]
            pe = 1.0 * v.T[1]
            n = len(mttf)

            se_mttf = scipy.stats.sem(mttf)
            se_pe = scipy.stats.sem(pe)

            h_mttf = se_mttf * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
            h_pe = se_pe * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)

            conf_data[k] = np.array([h_mttf, h_pe])

        self.confidence_interval_data = conf_data

        return conf_data

    def pareto_front(self, use_objectives=np.array([1.0, 1.0, 1.0])):
        print("USE_OBJECTIVES", use_objectives)
        # if self.pareto_front_data is not None:
        #     return self.pareto_front_data

        for k, v in self.means().items():
            k.fitness.values = tuple(v * use_objectives)
            # print(tuple(v * use_objectives))

        front = sortNondominated(self.data.keys(), 100, first_front_only=True)[0]
        self.pareto_front_data = front

        return front


if __name__ == "__main__":
    analysis = Analysis()

    print("Obtaining mean_data")
    mean_data = np.asarray(list(analysis.means().values()))

    print("Obtaining Pareto front")
    front = np.array([individual.fitness.values for individual in analysis.pareto_front])
    front_data = front.T[:3]
    print(front_data)

    mean_data.delete(front)


    # print("get_front")
    # pareto_front = analysis.get_pareto_front()
    #
    # print("second time mean")
    # data = analysis.get_mean()
