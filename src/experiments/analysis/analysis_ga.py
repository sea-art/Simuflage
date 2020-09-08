import pickle

import numpy as np
from deap.tools import sortNondominated


class AnalysisGA:
    def __init__(self):
        print("Reading data")
        loaded_data = pickle.load(open("out/pickles/logbooks/bestcands_ds1.p", "rb"))

        self.data = {}

        self.pareto_front_data = None

        print("Formatting data")
        for run in loaded_data:
            for k, v in run.items():
                self.data[k] = v

    def pareto_front(self, use_objectives=np.array([1.0, 1.0, 1.0])):
        for k, v in self.data.items():
            k.fitness.values = tuple(v * use_objectives)

        front = sortNondominated(self.data.keys(), 10, True)[0]
        self.pareto_front_data = front

        return front