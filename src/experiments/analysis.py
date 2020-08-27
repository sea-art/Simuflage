import multiprocessing
import pickle
import numpy as np
import scipy.stats
from random import shuffle
import itertools
import math


from deap import creator, base
from deap.tools import sortNondominated, selNSGA2

from DSE.exploration.GA import Chromosome
# from DSE.exploration.algorithm import *

weights = (1.0, -1.0, -1.0)

# creator.create("FitnessDSE_mcs", base.Fitness, weights=weights)
# creator.create("Individual_mcs", Chromosome, fitness=creator.FitnessDSE_mcs)


class AnalysisMCS:
    def __init__(self, dps_file_names=np.array(["dps2.p"]),
                       samples_file_names=np.array(["samples2.p"])):
        self.data = {}

        for idx in range(len(dps_file_names)):
            print("Reading data from", samples_file_names[idx])
            dps = pickle.load(open("out/pickles/" + dps_file_names[idx], "rb"))
            samples = pickle.load(open("out/pickles/" + samples_file_names[idx], "rb"))

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


class AnalysisGA:
    def __init__(self):
        print("Reading data")
        loaded_data = pickle.load(open("out/pickles/bestcands_ds1.p", "rb"))

        self.data = {}

        self.pareto_front_data = None

        print("Formatting data")
        for run in loaded_data:
            for k, v in run.items():
                self.data[k] = v

    def pareto_front(self, use_objectives=np.array([1.0, 1.0, 1.0])):
        for k, v in self.data.items():
            k.fitness.values = tuple(v * use_objectives)

        front = selNSGA2(self.data.keys(), 100)
        self.pareto_front_data = front

        return front


class Individual:
    def __init__(self, individual, samples):
        """

        :param individual: As ingle Individual_mcs object
        :param samples: A list of sampled values for this individual
        """
        self.individual = individual
        self.samples = samples
        np.random.shuffle(self.samples)
        self.mean = tuple(np.mean(samples, axis=0))

    def __repr__(self):
        return str(self.individual)

    def mcs(self, nr_samples, average=True):
        samples = self.samples[np.random.choice(self.samples.shape[0], nr_samples, replace=False)]

        return tuple(np.mean(samples, axis=0)) if average else samples


def update_empirical_mean(vector, ui, N):
    """

    @param vector: new reward vector
    @param ui: current empirical mean
    @param N: number of samples
    @return: new empirical mean vector
    """
    for i in range(len(vector)):
        ui[i] = ui[i] + (vector[i] - ui[i]) / N

    return ui


class Dps:
    def __init__(self, individuals, samples, to_sel=50):
        self.individuals = [Individual(k, v) for k, v in zip(individuals, samples)]
        self.dps = [individual.individual for individual in self.individuals]  # the list of deap Individual objects
        self.to_sel = to_sel

        self.ref_sel, self.ref_sel_idcs = self._get_ref_selection()

        all_samples = np.array(list(itertools.chain(*samples)))
        self.norm_vals = np.max(all_samples, axis=0)

    def _get_ref_selection(self):
        for dp in self.individuals:
            dp.individual.fitness.values = dp.mean

        selected = list(itertools.chain(*sortNondominated(self.dps, len(self.dps) // 2)))
        selected_idx = [list(self.dps).index(dp) for dp in selected]

        return selected, selected_idx

    def _add_confidence_interval(self, N, A_star_len, subtract=False):
        n = len(self.dps)
        ci = [math.sqrt(2 * math.log(sum(N) * (3 * A_star_len) ** (1 / 4)) / N[i]) for i in range(n)]

        for i in range(n):
            a, b, c = self.dps[i].fitness.values
            if not subtract:
                self.dps[i].fitness.values = (a + ci[i], b + ci[i], c + ci[i])
            else:
                self.dps[i].fitness.values = (a - ci[i], b - ci[i], c - ci[i])

    def _delta_pk_ij(self, ui, A, f_p, p):
        """

        @param ui: empricical means (list of reward vectors)
        @param A: Non rejected arms
        @param f_p: scalarization function
        @param p: how many arms to still accept
        @return:
        """
        t_ui = [(i, f_p(ui[i])) for i in range(len(ui))]
        sorted_indices = [i for (i, val) in sorted(t_ui, key=lambda x: x[1], reverse=True) if i in A]

        i_star_up = sorted_indices[p]
        i_star_down = sorted_indices[p + 1]
        gaps = [0 for _ in range(len(t_ui))]

        for i in sorted_indices[:p]:
            gaps[i] = t_ui[i][1] - t_ui[i_star_down][1]

        for i in sorted_indices[p:]:
            gaps[i] = t_ui[i_star_up][1] - t_ui[i][1]

        assert np.any(np.array(gaps) > 0.0), (p, sorted_indices, [t_ui[z] for z in sorted_indices], f_p([1, 3, 9]))

        # return the index of the maximum gap and if this arm should be accepted
        j = gaps.index(max(gaps))

        return j, j == sorted_indices[0]

    def normalize(self, sample, invert=False):
        return sample * self.norm_vals if invert else sample / self.norm_vals

    def mcs(self, nr_samples, average=True):
        nr_samples = nr_samples // len(self.individuals)
        return np.array([i.mcs(nr_samples, average=average) for i in self.individuals])

    def ssar(self, S, nr_samples):
        accepted_arms = [set() for _ in range(len(S))]

        K = len(self.individuals)
        A_all = [list(range(K)) for _ in range(len(S))]
        A = list(range(K))
        P_i = [self.to_sel for _ in range(len(S))]

        N = [0 for _ in range(K)]
        ui = [[0 for _ in range(3)] for _ in range(K)]

        n_k = 0
        LOG_K = 1/2 + sum([1 / i for i in range(2, K + 1)])

        for k in range(1, K):
            n_k_prev = n_k
            n_k = math.ceil((1 / LOG_K) * (nr_samples - K) / (K + 1 - k))

            samples = int(n_k - n_k_prev)
            samples = int(np.ceil((samples * K) / len(A)))

            for i in A:
                for _ in range(samples):
                    N[i] += 1
                    reward_vector = self.individuals[i].mcs(1)
                    ui[i] = update_empirical_mean(self.normalize(reward_vector), ui[i], N[i])

            for i in range(len(S)):
                max_gap_idx, accepted = self._delta_pk_ij(ui, A_all[i], S[i], P_i[i] - len(accepted_arms[i]))
                A_all[i].remove(max_gap_idx)

                if accepted:  # Store the arms that are accepted by a function for this round
                    accepted_arms[i].add(max_gap_idx)

            A = set().union(*A_all)  # Updates A to only include any arm that has not yet been removed by an F_j

        # print("n:", 6 * nr_samples, "actual:", sum(N))

        return np.array([self.normalize(np.array(z), invert=True) for z in ui]), N

    def pucb(self, nr_samples):
        n = len(self.dps)

        simulators = {self.dps[i]: self.individuals[i] for i in range(n)}

        N = {self.dps[i]: 1 for i in range(n)}
        ui = {self.dps[i]: self.normalize(self.individuals[i].mcs(1)) for i in range(n)}

        for i in self.dps:
            i.fitness.values = ui[i]

        cur_samples = sum(N.values())

        while cur_samples < nr_samples:
            A_star = sortNondominated(self.dps, self.to_sel, first_front_only=True)[0]
            self._add_confidence_interval(list(N.values()), len(A_star))

            A_p = sortNondominated(self.dps, self.to_sel, first_front_only=True)[0]
            self._add_confidence_interval(list(N.values()), len(A_star), subtract=True)

            a = np.random.choice(A_p)

            N[a] += 1
            cur_samples += 1

            ui[a] = update_empirical_mean(self.normalize(simulators[a].mcs(1)), ui[a], N[a])
            a.fitness.values = ui[a]

        print("Total samples:", sum(list(N.values())))

        return np.array([self.normalize(ui[self.dps[i]], invert=True) for i in range(n)])

    def wrong_selected(self, nr_samples, eval_method, number=False):
        if eval_method == 'pucb':
            print("Using pucb")
            samples = self.pucb(nr_samples)
        elif eval_method == 'ssar':
            print("Using ssar")
            samples, N = self.ssar(S, nr_samples)
        else:
            print("Using MCS")
            samples = self.mcs(nr_samples)

        for i, v in zip(self.dps, samples):
            i.fitness.values = v

        selected = selNSGA2(self.dps, self.to_sel)
        selected_idcx = [self.dps.index(dp) for dp in selected]

        wrong_selected = [i for i in selected_idcx if i not in self.ref_sel_idcs]

        total_samples = sum(N) if eval_method == 'ssar' else nr_samples

        if number:
            return len(wrong_selected), total_samples

        return wrong_selected, total_samples


def get_all_weights(steps=10):
    """ Will create a list of all possible weight values for the scalarization functions.

    :type steps: int representing the stepsize (as 1 / steps)
    :return:
    """
    boxes = 3
    values = steps

    rng = list(range(values + 1)) * boxes
    ans = sorted(set(i for i in itertools.permutations(rng, boxes) if sum(i) == values))

    return list(map(lambda tup: (tup[0]/steps, tup[1]/steps, tup[2]/steps), ans))


def linear_scalarize(vector, w):
    return sum([vector[i] * w[i] for i in range(len(vector))])


def scalarized_lambda(w):
    return lambda vec: linear_scalarize(vec, w)


def wrong_sel_pucb(output, identifier, dps, samples):
    temp = []

    for i, dp in enumerate(dps[::33]):
        print(str(i) + "/100", "done")
        faults, _ = dp.wrong_selected(samples * 100, 'pucb', number=True)
        temp.append(faults)

    print(temp)

    output[(identifier, samples)] = sum(temp)


def wrong_sel_ssar(output, identifier, dps, samples):
    temp = {}

    for i, dp in enumerate(dps[::250]):
        print(str(i) + "/" + str(len(dps)), "done")
        nr_faults, N = dp.wrong_selected((samples * 100) // 10, 'ssar', number=True)
        temp[N] = nr_faults

    print(temp)

    output[(identifier, sum(list(temp.keys())) // len(temp.keys()))] = sum(list(temp.values()))


S = [scalarized_lambda(w) for w in get_all_weights() if w[2] != 1.0][::2]


def create_dps_set():
    """
    Usefull to obtain dictionary of 700 sets of 100 individuals with each 10.000 samples.
    :return:
    """
    dps_names = np.array(["dps2.p", "dps3.p", "dps4.p"])
    samples_names = np.array(["samples2.p", "samples3.p", "samples4.p"])

    analysis = AnalysisMCS(dps_names[:1], samples_names[:1])

    pop_size = 100

    keys = list(analysis.data.keys())
    values = list(analysis.data.values())

    print("Storing", len(keys), "in", len(keys) // pop_size, "chunks")

    dps = [Dps(keys[i:i + pop_size], values[i:i + pop_size]) for i in
           range(0, len(analysis.data.keys()) - pop_size + 1, pop_size)]

    print("Pickling data")
    pickle.dump(dps, open("out/pickles/working_dps.p", "wb"))

    return dps

if __name__ == "__main__":
    analysis = AnalysisGA()
    print(analysis.data)

    # print("Loading")
    # dps = pickle.load(open("out/pickles/working_dps.p", "rb"))
    #
    # wrong_sel = {}
    #
    # for i, sa in zip(range(26, 502, 25), [32, 55, 77, 100, 123, 145, 169, 190, 214, 236, 260, 281, 303, 328, 351, 374, 398, 418, 440, 466]):
    #     print("Finding wrong selected with", i, "samples per dp")
    #     with open("out/logs/ssar_wrong_selected1.csv", "a") as myfile:
    #         jobs = []
    #
    #         manager = multiprocessing.Manager()
    #         return_dict = manager.dict()
    #
    #         for j in range(20):
    #             jobs.append(multiprocessing.Process(target=wrong_sel_ssar,
    #                                                 args=(return_dict, j, dps, sa)))
    #
    #         for j in jobs:
    #             j.start()
    #
    #         for j in jobs:
    #             j.join()
    #
    #         print(return_dict)
    #
    #         total_incorrect = sum(list(return_dict.values()))
    #         wrong_sel[i] = total_incorrect
    #         myfile.write(str(i) + "," + str(wrong_sel[i]) + "\n")
    #         print("finished", i, ":", total_incorrect)
    #
    # print(wrong_sel)
    #
    # for k, v in wrong_sel.items():
    #     print(k, v)
