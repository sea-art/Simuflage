import pickle
import numpy as np

from deap.tools import sortNondominated


class AnalysisMCS:
    def __init__(self, dps_file_names=np.array(["dps2.p"]),
                       samples_file_names=np.array(["samples2.p"])):
        self.data = {}

        for idx in range(len(dps_file_names)):
            print("Reading data from", samples_file_names[idx])
            dps = pickle.load(open("out/pickles/samples/" + dps_file_names[idx], "rb"))
            samples = pickle.load(open("out/pickles/samples/" + samples_file_names[idx], "rb"))

            print("Formatting data from", samples_file_names[idx])

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
        # if self.pareto_front_data is not None:
        #     return self.pareto_front_data

        for k, v in self.means().items():
            k.fitness.values = tuple(v * use_objectives)

        front = sortNondominated(self.data.keys(), 100, first_front_only=True)[0]
        self.pareto_front_data = front

        return front
