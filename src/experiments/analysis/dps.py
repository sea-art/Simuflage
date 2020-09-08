import itertools
import math

import numpy as np
import scipy.spatial.distance as dist

from deap.benchmarks.tools import hypervolume
from deap.tools import sortNondominated, selNSGA2

from .individual import Individual
from .statics import update_empirical_mean, update_emperical_mean_set, scalarized_lambda, get_all_weights


class Dps:
    def __init__(self, individuals, samples, to_sel=50):
        self.individuals = [Individual(k, v) for k, v in zip(individuals, samples)]
        self.dps = [individual.individual for individual in self.individuals]  # the list of deap Individual objects
        self.to_sel = to_sel

        self.means = {individuals[i]: self.individuals[i].mean for i in range(len(individuals))}

        self.ref_sel, self.ref_sel_idcs = self._get_ref_selection()
        self.hvol_refpoint = np.array([-1, 600.0, 37.0])
        self.ref_hvol = hypervolume(self.ref_sel, self.hvol_refpoint)

        all_samples = np.array(list(itertools.chain(*samples)))
        self.norm_vals = np.max(all_samples, axis=0)

    def _get_ref_selection(self):
        for dp in self.individuals:
            dp.individual.fitness.values = dp.mean

        selected = list(itertools.chain(*sortNondominated(self.dps, len(self.dps) // 2)))
        # selected = selNSGA2(self.dps, len(self.dps) // 2)
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
        accepted_arms = [[] for _ in range(len(S))]

        K = len(self.individuals)
        A_all = [list(range(K)) for _ in range(len(S))]
        A = list(range(K))
        P_i = [self.to_sel for _ in range(len(S))]

        order_removed = []

        nr_samples = nr_samples

        N = [0 for _ in range(K)]
        ui = [[0 for _ in range(3)] for _ in range(K)]

        n_k = 0  # NOTE: If n_k is set to negative number, that number will be spent towards initial sample
                 #       which can be taken of the initial budget.
        LOG_K = 1/2 + sum([1 / i for i in range(2, K + 1)])

        total_accepted = 0
        total_rejected = 0

        for k in range(1, K):
            n_k_prev = n_k
            n_k = math.ceil((1 / LOG_K) * ((nr_samples - K) / (K + 1 - k)))

            samples = int(n_k - n_k_prev)
            # samples = int(np.round(samples * (K / len(A))))

            for i in A:
                if samples > 0:
                    reward_vector = self.individuals[i].mcs(samples)
                    # print(reward_vector, S[0](reward_vector))
                    ui[i] = update_emperical_mean_set(self.normalize(reward_vector), ui[i], N[i], samples)
                    N[i] += samples


            for i in range(len(S)):
                max_gap_idx, accepted = self._delta_pk_ij(ui, A_all[i], S[i], P_i[i] - len(accepted_arms[i]))
                A_all[i].remove(max_gap_idx)

                if accepted:  # Store the arms that are accepted by a function for this round
                    accepted_arms[i].append(max_gap_idx)
                    total_accepted += 1
                else:
                    total_rejected += 1

            A = set().union(*A_all)  # Updates A to only include any arm that has not yet been removed by an F_j

        print(N)

        accpted_idcx = list(itertools.chain(*accepted_arms))

        return np.array([self.normalize(np.array(z), invert=True) for z in ui]), N, order_removed, accpted_idcx

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

        # print("\nTotal samples:", sum(list(N.values())))

        return np.array([self.normalize(ui[self.dps[i]], invert=True) for i in range(n)])

    def distance_selections(self, nr_samples, eval_method, metric='eucl_dist'):
        """

        :param nr_samples:
        :param eval_method:
        :param distance: Choice of ['eucl_dist', 'hypervolume']
        :return:
        """
        if eval_method == 'pucb':
            samples = self.pucb(nr_samples)
        elif eval_method == 'ssar':
            samples, N, order, accepted_idcx = self.ssar(S, nr_samples)
        else:
            samples = self.mcs(nr_samples)

        for i, v in zip(self.dps, samples):
            i.fitness.values = v

        total_samples = sum(N) if eval_method == 'ssar' else nr_samples

        selected = selNSGA2(self.dps, self.to_sel)
        reference = self.ref_sel

        if metric == 'eucl_dist':
            distance = np.min(dist.cdist(np.array([i.fitness.wvalues for i in selected]),
                                         np.array([i.fitness.wvalues for i in reference])), axis=1)
            distance = np.mean(distance)
        elif metric == 'hypervolume':
            distance = math.fabs(self.ref_hvol - hypervolume(selected, self.hvol_refpoint))

        return distance, total_samples

    def wrong_selected(self, nr_samples, eval_method, number=False):
        S = [scalarized_lambda(w) for w in get_all_weights() if w[0] != 0.0]

        if eval_method == 'pucb':
            samples = self.pucb(nr_samples)
        elif eval_method == 'ssar':
            samples, N, order, accepted_idcx = self.ssar(S, nr_samples)
        else:
            samples = self.mcs(nr_samples)

        # cor_order = [i in self.ref_sel_idcs for i in order]
        # accepted_idcx = [i in accepted_idcx for i in order]
        # N_used = [N[i] for i in order]
        # sample_diff = [np.array(samples[i]) - np.array(self.dps[i].fitness.values) for i in order]
        # data = list(zip(order, N_used, cor_order, accepted_idcx, sample_diff))
        #
        # df = pd.DataFrame(data, columns=["idx", "samples", "accepted", "in ref", "sampled_diff"])
        # df.set_index('idx', inplace=True)

        for i, v in zip(self.dps, samples):
            i.fitness.values = v

        selected = selNSGA2(self.dps, self.to_sel)
        selected_idcx = [self.dps.index(dp) for dp in selected]

        wrong_selected = [i for i in selected_idcx if i not in self.ref_sel_idcs]

        total_samples = sum(N) if eval_method == 'ssar' else nr_samples

        if number:
            return len(wrong_selected), total_samples

        return wrong_selected, total_samples