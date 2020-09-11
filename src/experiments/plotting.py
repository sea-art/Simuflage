import itertools

import matplotlib.colors
from deap.tools import sortNondominated
from matplotlib.ticker import ScalarFormatter

from DSE.exploration.GA.algorithm import GA, initialize_sesp
from design.mapping import first_fit, best_fit, next_fit, worst_fit
from experiments import AnalysisMCS
from mpl_toolkits.mplot3d import Axes3D

import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import scipy.stats as st
import scipy.spatial.distance as dst
import pandas as pd
from scipy.spatial import distance
import pickle

from experiments.analysis import AnalysisGA, Individual, Dps


def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n - 1)
    return h


class PlottingSimulator:
    def __init__(self):
        self.fontsize = 18
        self.ticksize = 14
        self.data_samples = pickle.load(open("out/pickles/plotSimdp.p", "rb"))
        self.data_many = pickle.load(open("out/pickles/large_dataset.p", "rb"))

        # print(len(self.data_samples))
        # self.means_per_policy_and_appmapping()

        self.plot_all()

    def plot_all(self):
        self.plot_histograms_of_all_samples(self.data_samples, "sample_dataset")
        self.plot_histograms_of_all_samples(self.data_many, "large_dataset")
        self.means_as_components_increases()
        self.means_per_policy_and_appmapping()

    def plot_3D_objective_space(self):
        analyser = AnalysisMCS(dps_file_names=["dps2.p"], samples_file_names=["samples2.p"])
        means = np.array(list(analyser.means().values()))

        mttf = means.T[0] / (24 * 365)
        pe = means.T[1]
        size = means.T[2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(mttf, pe, size)
        ax.set_xlabel('MTTF in years')
        ax.set_ylabel('Energy consumption')
        ax.set_zlabel('Size')
        ax.view_init(30, 45)
        plt.show()
        fig.savefig("out/plots/3d_objective_space_example_a1.pdf")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(mttf, pe, size)
        ax.set_xlabel('MTTF in years')
        ax.set_ylabel('Energy consumption')
        ax.set_zlabel('Size')
        ax.view_init(30, 135)
        plt.show()
        fig.savefig("out/plots/3d_objective_space_example_a2.pdf")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(mttf, pe, size)
        ax.set_xlabel('MTTF in years')
        ax.set_ylabel('Energy consumption')
        ax.set_zlabel('Size')
        ax.view_init(30, 215)
        plt.show()
        fig.savefig("out/plots/3d_objective_space_example_a3.pdf")

    def means_per_policy_and_appmapping(self):
        policies = np.array(['most', 'least', 'random'])
        app_mappings = ["first_fit", "best_fit", "next_fit", "worst_fit"]

        combs = list(itertools.product(policies, app_mappings))

        app_mapping_plot_names = np.array(['First fit', 'Best fit', 'Next fit', "Worst fit"])

        xs = np.arange(4)
        values_mttf, errs_mttf = {}, {}
        values_pe, errs_pe = {}, {}

        for policy, app_map in combs:
            dps = [k for k in list(self.data_many.keys()) if k.genes[3].policy == policy and k.genes[2].map_func.__name__ == app_map]
            values_mttf[(policy, app_map)] = (np.mean([np.mean(self.data_many[dp].T[0]) for dp in dps]) / (24 * 365))
            errs_mttf[(policy, app_map)] = (np.mean([np.std(self.data_many[dp].T[0]) for dp in dps]) / (24 * 365))

            values_pe[(policy, app_map)] = (np.mean([np.mean(self.data_many[dp].T[1]) for dp in dps]))
            errs_pe[(policy, app_map)] = (np.mean([np.std(self.data_many[dp].T[1]) for dp in dps]))

        mttf_most, mttf_most_err = list(values_mttf.values())[:4], list(errs_mttf.values())[:4]
        mttf_least, mttf_least_err = list(values_mttf.values())[4:8], list(errs_mttf.values())[4:8]
        mttf_random, mttf_random_err = list(values_mttf.values())[8:], list(errs_mttf.values())[8:]

        pe_most, pe_most_err = list(values_pe.values())[:4], list(errs_pe.values())[:4]
        pe_least, pe_least_err = list(values_pe.values())[4:8], list(errs_pe.values())[4:8]
        pe_random, pe_random_err = list(values_pe.values())[8:], list(errs_pe.values())[8:]

        fig = plt.figure()
        ax = plt.subplot(111)
        w = 0.3
        ax.bar(xs - w, mttf_random, width=w, yerr=mttf_random_err, color='c', edgecolor='k', alpha=0.65, label="Random")
        ax.bar(xs, mttf_most, width=w, yerr=mttf_most_err, color='r', edgecolor='k', alpha=0.65, label="Most-slack first")
        ax.bar(xs + w, mttf_least, width=w, yerr=mttf_least_err, color='y', edgecolor='k', alpha=0.65, label="Least-slack first")
        plt.xticks(xs, app_mapping_plot_names, fontsize=self.ticksize)
        plt.legend(loc='center right', fontsize=self.fontsize)
        plt.ylabel("MTTF in years", fontsize=self.fontsize)
        plt.yticks(fontsize=self.ticksize)
        plt.show()
        plt.tight_layout()
        fig.savefig("out/plots/simulator/policy_and_app_mapping_mttf.pdf", bbox_inches='tight')

        fig = plt.figure()
        ax = plt.subplot(111)
        w = 0.3
        ax.bar(xs - w, pe_random, width=w, yerr=pe_random_err, color='c', edgecolor='k', alpha=0.65, label="Random")
        ax.bar(xs, pe_most, width=w, yerr=pe_most_err, color='r', edgecolor='k', alpha=0.65, label="Most-slack first")
        ax.bar(xs + w, pe_least, width=w, yerr=pe_least_err, color='y', edgecolor='k', alpha=0.65, label="Least-slack first")
        plt.xticks(xs, app_mapping_plot_names, fontsize=self.ticksize)
        plt.legend(loc='center right', fontsize=self.fontsize)
        plt.ylabel("Energy consumption", fontsize=self.fontsize)
        plt.yticks(fontsize=self.ticksize)
        plt.show()
        plt.tight_layout()
        fig.savefig("out/plots/simulator/policy_and_app_mapping_pe.pdf", bbox_inches='tight')

    def means_as_components_increases(self):
        dps_set = [[k for k in list(self.data_samples.keys()) if len(k.genes[0]) == i] for i in range(2, 21)]

        xs = list(range(2, 21))
        ys_mttf = []
        ys_pe = []
        ys_size = []

        er_mttf = []
        er_pe = []
        er_size = []

        for i, dps in zip(range(2, 21), dps_set):

            ys_mttf.append(np.mean([np.mean(self.data_samples[dp].T[0]) for dp in dps]) / (24 * 365))
            ys_pe.append(np.mean([np.mean(self.data_samples[dp].T[1]) for dp in dps]))
            ys_size.append(np.mean([np.mean(self.data_samples[dp].T[2]) for dp in dps]))

            er_mttf.append(np.mean([np.std(self.data_samples[dp].T[0]) for dp in dps]) / (24 * 365))
            er_pe.append(np.mean([np.std(self.data_samples[dp].T[1]) for dp in dps]))
            er_size.append(np.mean([np.std(self.data_samples[dp].T[2]) for dp in dps]))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.15)
        ax.bar(xs, ys_mttf, yerr=er_mttf, color='c', edgecolor='k', alpha=0.65)
        plt.xlabel("Number of components", fontsize=self.fontsize)
        plt.ylabel("MTTF in years", fontsize=self.fontsize)
        plt.xticks(fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.show()
        fig.savefig("out/plots/simulator/mttf_as_components_increase.pdf", bbox_inches='tight')

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.15)
        ax.bar(xs, ys_pe, yerr=er_pe, color='y', edgecolor='k', alpha=0.65)
        plt.xlabel("Number of components", fontsize=self.fontsize)
        plt.ylabel("Energy consumption", fontsize=self.fontsize)
        plt.xticks(fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.show()
        fig.savefig("out/plots/simulator/pe_as_components_increase.pdf", bbox_inches='tight')

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.15)
        ax.bar(xs, ys_size, yerr=er_size, color='g', edgecolor='k', alpha=0.65)
        plt.xlabel("Number of components", fontsize=self.fontsize)
        plt.ylabel("Grid size", fontsize=self.fontsize)
        plt.xticks(fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.show()
        fig.savefig("out/plots/simulator/size_as_components_increase.pdf", bbox_inches='tight')

    def plot_histograms_of_all_samples(self, dataset, dataset_name):
        samples = np.stack(np.array(list(dataset.values())))

        all_samples = samples.reshape((-1, 3))

        mttf = all_samples.T[0] / (24 * 365)
        pe = all_samples.T[1]

        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.15)
        plt.hist(mttf, density=True, bins=30, color='c', edgecolor='k', alpha=0.65, label="Samples")
        mn, mx = plt.xlim()
        plt.xlim(mn, mx)
        plt.axvline(mttf.mean(), color='k', linestyle='dashed', linewidth=2)
        min_ylim, max_ylim = plt.ylim()
        plt.text(mttf.mean() * 0.30, max_ylim * 0.82, 'Mean: {:.2f}'.format(mttf.mean()), fontsize=self.fontsize - 2)
        plt.xlabel("MTTF in years", fontsize=self.fontsize)
        plt.ylabel("Probability", fontsize=self.fontsize)
        plt.xticks(fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.show()
        f.savefig("out/plots/simulator/hist_mttf_{}.pdf".format(dataset_name))

        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.15)
        plt.hist(pe, 20, density=True, color='y', edgecolor='k', alpha=0.65, label="Samples")
        mn, mx = plt.xlim()
        plt.xlim(mn, mx)
        plt.axvline(pe.mean(), color='k', linestyle='dashed', linewidth=2)
        min_ylim_pe, max_ylim_pe = plt.ylim()
        plt.text(pe.mean() * 1.07, max_ylim_pe * 0.94, 'Mean: {:.2f}'.format(pe.mean()), fontsize=self.fontsize - 2)
        plt.xlabel("Energy consumption", fontsize=self.fontsize)
        plt.ylabel("Probability", fontsize=self.fontsize)
        plt.xticks(fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.show()
        f.savefig("out/plots/simulator/hist_pe_{}.pdf".format(dataset_name))

    def plot_histograms_of_samples(self, dp, filename1, filename2):
        samples = np.array(self.data_samples[dp])

        mttf = samples.T[0] / (24 * 365)
        pe = samples.T[1]

        f = plt.figure()
        plt.hist(mttf, density=True, bins=30, color='c', edgecolor='k', alpha=0.65, label="Samples")
        mn, mx = plt.xlim()
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, 301)
        kde = st.gaussian_kde(mttf)
        plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
        plt.axvline(mttf.mean(), color='k', linestyle='dashed', linewidth=2)
        min_ylim, max_ylim = plt.ylim()
        plt.text(mttf.mean() * 1.06, max_ylim * 0.9, 'Mean: {:.2f}'.format(mttf.mean()), fontsize=self.fontsize - 2)
        plt.xlabel("MTTF in years", fontsize=self.fontsize)
        plt.ylabel("Probability", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize, loc='upper left')
        plt.show()
        f.savefig(filename1, bbox_inches='tight')

        f = plt.figure()
        plt.hist(pe, 20, density=True, color='y', edgecolor='k', alpha=0.65, label="Samples")
        mn, mx = plt.xlim()
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, 301)
        kde = st.gaussian_kde(pe)
        plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
        plt.axvline(pe.mean(), color='k', linestyle='dashed', linewidth=2)
        min_ylim_pe, max_ylim_pe = plt.ylim()
        plt.text(pe.mean() * 0.93, max_ylim_pe * 0.90, 'Mean: {:.2f}'.format(pe.mean()), fontsize=self.fontsize - 2)
        plt.xlabel("Energy consumption", fontsize=self.fontsize)
        plt.ylabel("Probability", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize)
        plt.show()
        f.savefig(filename2, bbox_inches='tight')


class PlottingEvaluation:
    def __init__(self):
        self.ticksize = 14
        self.data_samples = pickle.load(open("out/pickles/plotSimdp.p", "rb"))
        self.data_many = pickle.load(open("out/pickles/large_dataset.p", "rb"))

        self.fontsize = 16
        self.grid_alpha = 0.1
        self.plot_folder = "out/plots/simulator/"
        #
        self.higher_wl_lower_mttf()
        self.higher_capacity_higher_mttf()
        self.distance_increases_mttf()
        self.higher_capacity_higher_usage()

    def higher_wl_lower_mttf(self):
        dataset = self.data_samples

        workload = np.array([80 / np.sum(x.genes[0].values) for x in list(dataset.keys())])
        mttf = np.array([np.mean(z, axis=0) for z in dataset.values()]).T[0] / (24 * 365)

        ys = mttf[workload.argsort()]
        xs = np.sort(workload) * 100

        coeff = np.polyfit(xs, ys, 6)
        fit = np.poly1d(coeff)

        f = plt.figure()

        # plt.grid(True, alpha=self.grid_alpha)
        # plt.title("Average lifetime at different workload", fontsize=self.fontsize)
        plt.scatter(xs, ys, marker='+', c='c', alpha=0.7, label="Sample")
        plt.plot(xs, fit(xs), c='b', label="Fit")

        # plt.xlim(1.0, 5.0)
        plt.xlabel("Workload in percentage", fontsize=self.fontsize)
        plt.ylabel("MTTF in years", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize)
        plt.xticks(fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.tight_layout()
        plt.show()
        f.savefig(self.plot_folder + "higher_wl_lower_mttf" + ".pdf")

    def higher_capacity_higher_mttf(self):
        dataset = self.data_samples

        sum_capacities = np.array([np.sum(x.genes[0].values) for x in list(dataset.keys())])
        usages = np.array([np.mean(z, axis=0) for z in dataset.values()]).T[0] / (24 * 365)

        ys = usages[sum_capacities.argsort()]
        xs = np.sort(sum_capacities)

        coeff = np.polyfit(xs, ys, 2)
        fit = np.poly1d(coeff)

        f = plt.figure()

        # plt.grid(True, alpha=self.grid_alpha)
        # plt.title("Average lifetime at total computational capacity", fontsize=self.fontsize)
        plt.scatter(xs, ys, marker='+', c='c', alpha=0.7, label="Sample")
        plt.plot(xs, fit(xs), c='b', label="Fit")

        # plt.xlim(1.0, 5.0)
        plt.xlabel("Total computational capacity", fontsize=self.fontsize)
        plt.ylabel("MTTF in years", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize)
        plt.xticks(fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.tight_layout()
        plt.show()
        f.savefig(self.plot_folder + "higher_capacity_higher_mttf" + ".pdf")

    def higher_capacity_higher_usage(self):

        dataset = self.data_samples

        sum_capacities = np.array([np.sum(x.genes[0].values) for x in list(dataset.keys())])
        usages = np.array([np.mean(z, axis=0) for z in dataset.values()]).T[1]

        ys = usages[sum_capacities.argsort()]
        xs = np.sort(sum_capacities)

        coeff = np.polyfit(xs, ys, 1)
        fit = np.poly1d(coeff)

        f = plt.figure()

        # plt.grid(True, alpha=self.grid_alpha)
        # plt.title("Energy consumption at total computational capacity", fontsize=self.fontsize)
        plt.scatter(xs, ys, marker='+', c='y', alpha=0.7, label="Sample")
        plt.plot(xs[xs <= 2100], fit(xs[xs <= 2100]), c='b', label="Fit")

        # plt.xlim(1.0, 5.0)
        plt.xlabel("Total computational capacity", fontsize=self.fontsize)
        plt.ylabel("Energy consumption", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize)
        plt.xticks(fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.tight_layout()
        plt.show()
        f.savefig(self.plot_folder + "higher_capacity_higher_usage" + ".pdf")

    def distance_increases_mttf(self):
        dataset = self.data_many

        locs = np.array([np.array(x.genes[1].locations) for x in list(dataset.keys())])

        avg_distances = np.array([sum(dst.pdist(fp, 'cityblock')) / len(dst.pdist(fp, 'cityblock')) for fp in locs])

        all_xs = avg_distances
        all_ys = np.array([np.mean(z, axis=0) for z in dataset.values()]).T[0]

        xs_locs = all_xs <= 6.0

        all_xs = all_xs[xs_locs]
        all_ys = all_ys[xs_locs]

        all_ys = all_ys[all_xs.argsort()]
        all_xs = np.sort(all_xs)

        unique_xs = np.unique(all_xs)

        ys = []

        for x in unique_xs:
            idcx = all_xs == x
            ys.append(np.mean(all_ys[idcx]) / (24 * 365))

        coeff = np.polyfit(unique_xs, ys, 3)
        fit = np.poly1d(coeff)

        f = plt.figure()

        # plt.grid(True, alpha=self.grid_alpha)
        # plt.title("MTTF at average distances between CPUs on grid", fontsize=self.fontsize)
        plt.plot(unique_xs, ys, c='c', alpha=0.7, label="Sampled")
        plt.plot(unique_xs[unique_xs <= 5.8], fit(unique_xs[unique_xs <= 5.8]), c='b', label="Fitted")

        # plt.xlim(1.0, 5.0)
        plt.xlabel("Average Manhattan distance between points", fontsize=self.fontsize)
        plt.ylabel("MTTF in years", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize)
        plt.xticks(fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.tight_layout()
        plt.show()
        f.savefig(self.plot_folder + "distance_increases_mttf" + ".pdf")

    def wrongly_selected_per_eval_method(self):
        mcs_sample_input = list(range(26, 502, 25))
        mcs_wrongly_selected = np.array([18992, 14265, 11989, 10375, 9391, 8607, 7829, 7340, 7000,
                                         6567, 6270, 5932, 5647, 5536, 5073, 4968, 4634, 4508, 4204, 4042])

        ssar_sample_input = list(range(26, 427, 25))
        ssar_wrongly_selected = [8240, 11500, 8740, 9900, 8340, 9820, 8600, 9040, 8740, 8760, 7740, 8240,
                                 8680, 7780, 7980, 8160, 7280]

        f, ax = plt.subplots()
        ax.plot(mcs_sample_input, mcs_wrongly_selected, label="MCS")
        ax.plot(ssar_sample_input, ssar_wrongly_selected, label="sSAR")
        plt.legend(fontsize=self.fontsize)
        plt.show()
        f.savefig()

    def ci_as_samples_increase(self):
        dps = pickle.load(open("out/pickles/data_many_samples_as_dps.p", "rb"))

        mean_cis_mttf, std_cis_mttf = [], []
        mean_cis_pe, std_cis_pe = [], []

        xs = list(range(11, 3012, 100))

        for i in xs:
            print("At:", i, "samples")
            samples = np.array([ind.mcs(i, False) for ind in dps.individuals])
            cis_mttf = confidence_interval(samples.T[0])
            cis_pe = confidence_interval(samples.T[1])

            mean_cis_mttf.append(np.mean(cis_mttf) / (24 * 365))
            std_cis_mttf.append(np.std(cis_mttf) / (24 * 365))
            mean_cis_pe.append(np.mean(cis_pe))
            std_cis_pe.append(np.std(cis_pe))

        mean_cis_mttf = np.array(mean_cis_mttf)
        std_cis_mttf = np.array(std_cis_mttf)
        mean_cis_pe = np.array(mean_cis_pe)
        std_cis_pe = np.array(std_cis_pe)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.15)
        ax.plot(xs, mean_cis_mttf, color='c', label="Mean")
        ax.fill_between(xs, mean_cis_mttf - std_cis_mttf, mean_cis_mttf + std_cis_mttf,
                        color='c', alpha=0.2, label="SD")
        plt.ylim(0, 1.1)
        plt.xlabel("Number of samples", fontsize=self.fontsize)
        plt.ylabel("95% confidence interval", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize)
        plt.show()
        fig.savefig("out/plots/simulator/95ci_over_samples_mttf.pdf")

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.15)
        ax.plot(xs, mean_cis_pe, color='y', label="Mean")
        ax.fill_between(xs, mean_cis_pe - std_cis_pe, mean_cis_pe + std_cis_pe, color='y', alpha=0.2, label="SD")
        plt.ylim(-0.1, 3.6)
        plt.xlabel("Number of samples", fontsize=self.fontsize)
        plt.ylabel("95% confidence interval", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize)
        plt.show()
        fig.savefig("out/plots/simulator/95ci_over_samples_pe.pdf")


class PlottingGA:
    def __init__(self):
        self.logbooks_100mcs = pickle.load(open("out/pickles/logbooks/hypervolume/logbooks_100.p", "rb"))
        self.logbooks_50mcs = pickle.load(open("out/pickles/logbooks/hypervolume/logbooks_50.p", "rb"))
        self.logbooks_10mcs = pickle.load(open("out/pickles/logbooks/hypervolume/logbooks_10.p", "rb"))
        self.logbooks_1mcs = pickle.load(open("out/pickles/logbooks/hypervolume/logbooks_1.p", "rb"))
        self.bestcands = pickle.load(open("out/pickles/logbooks/old/bestcands_ds1.p", "rb"))

        self.logbooks_100ssar = pickle.load(open("out/pickles/logbooks/logbooks_ssar28.p", "rb"))
        self.logbooks_50ssar = pickle.load(open("out/pickles/logbooks/logbooks_ssar14.p", "rb"))
        self.logbooks_10ssar = pickle.load(open("out/pickles/logbooks/logbooks_ssar4.p", "rb"))

        # self.logbooks_30pucb = pickle.load(open("out/pickles/logbooks/logbooks_pucb30.p", "rb"))
        # self.logbooks_20pucb = pickle.load(open("out/pickles/logbooks/logbooks_pucb20.p", "rb"))
        # self.logbooks_10ssar = pickle.load(open("out/pickles/logbooks/logbooks_pucb30.p", "rb"))

        self.ticksize = 12
        self.fontsize = 16
        self.grid_alpha = 0.1
        self.plot_folder = "out/plots/ga/"
        plt.rc('grid', linestyle="--", color='black')

        # sSAR
        frames100_ssar = [pd.DataFrame(book) for book in self.logbooks_100ssar]
        frames50_ssar = [pd.DataFrame(book) for book in self.logbooks_50ssar]
        frames10_ssar = [pd.DataFrame(book) for book in self.logbooks_10ssar]
        self.df_ssar100 = pd.concat(frames100_ssar)
        self.df_ssar50 = pd.concat(frames50_ssar)
        self.df_ssar10 = pd.concat(frames10_ssar)
        self.dfs_ssar = {10: self.df_ssar10, 50: self.df_ssar50, 100: self.df_ssar100}

        # MCS
        frames100_mcs = [pd.DataFrame(book) for book in self.logbooks_100mcs]
        frames50_mcs = [pd.DataFrame(book) for book in self.logbooks_50mcs]
        frames10_mcs = [pd.DataFrame(book) for book in self.logbooks_10mcs]
        frames1_mcs = [pd.DataFrame(book) for book in self.logbooks_1mcs]
        self.df_mcs100 = pd.concat(frames100_mcs)
        self.df_mcs50 = pd.concat(frames50_mcs)
        self.df_mcs10 = pd.concat(frames10_mcs)
        self.df_mcs1 = pd.concat(frames1_mcs)

        self.dfs_mcs = {1: self.df_mcs1, 10: self.df_mcs10, 50: self.df_mcs50, 100: self.df_mcs100}

        # print(self.df100.columns.values)
        #
        # means100 = np.array([np.mean(self.df100.loc[self.df100['gen'] == i, 'mean']) for i in range(0, 51)])
        # means_sampled100 = np.array([np.mean(self.df100.loc[self.df100['gen'] == i, 'sampled_mean']) for i in range(0, 51)])
        #
        # means50 = np.array([np.mean(self.df50.loc[self.df50['gen'] == i, 'mean']) for i in range(0, 51)])
        # means_sampled50 = np.array([np.mean(self.df50.loc[self.df50['gen'] == i, 'sampled_mean']) for i in range(0, 51)])
        #
        # means10 = np.array([np.mean(self.df10.loc[self.df10['gen'] == i, 'mean']) for i in range(0, 51)])
        # means_sampled10 = np.array([np.mean(self.df10.loc[self.df10['gen'] == i, 'sampled_mean']) for i in range(0, 51)])
        #
        # means1 = np.array([np.mean(self.df1.loc[self.df1['gen'] == i, 'mean']) for i in range(0, 51)])
        # means_sampled1 = np.array([np.mean(self.df1.loc[self.df1['gen'] == i, 'sampled_mean']) for i in range(0, 51)])
        #
        # print("Actual\t\t\t\t\t\t\t\t\t\t\t", "Sampled")
        # for a, b, c, d in zip(means1, means_sampled1, means100, means_sampled100):
        #     print(a, b, " \t", c, d)
        #
        # # print(means100)
        # # print(means_sampled100)
        #
        # exit(1)

        # self.plot_all()
        self.plot_ssar_vs_mcs_comparisons()

    def plot_all(self):
        self.plot_means_over_gens()
        self.plot_best_over_gens()
        self.plot_other_ga_info()
        self.plot_samples_comparisons()

    def plot_means_over_gens(self):
        self._plot_mean_mttf_over_gens()
        self._plot_mean_pe_over_gens()
        self._plot_mean_size_over_gens()

    def plot_best_over_gens(self):
        self._plot_best_mttf_over_gens()
        self._plot_best_pe_over_gens()
        self._plot_best_size_over_gens()

    def plot_other_ga_info(self):
        self._plot_average_elitism_over_gens()
        self._plot_offspring_and_mutations_over_gens()

    def plot_samples_comparisons(self):
        self._plot_mean_mttf_different_samples()
        self._plot_mean_pe_different_samples()
        self._plot_mean_size_different_samples()

        self._plot_best_mttf_different_samples()
        self._plot_best_pe_different_samples()
        self._plot_best_size_different_samples()

        self._plot_elitism_different_samples()
        self._plot_distance_over_gens()

    def plot_ssar_vs_mcs_comparisons(self):
        self._plot_objectives_ssar_vs_mcs(column_name='mean')
        self._plot_objectives_ssar_vs_mcs(column_name='best')
        self._plot_distance_ssar_vs_mcs()

    def _plot_distance_ssar_vs_mcs(self):
        xs = np.arange(0, 51)

        for samples in self.dfs_ssar.keys():
            df_ssar = self.dfs_ssar[samples]
            df_mcs = self.dfs_mcs[samples]

            distance_ssar = np.array([df_ssar.loc[df_ssar['gen'] == i, 'distance'] for i in xs])
            distance_mcs = np.array([df_mcs.loc[df_mcs['gen'] == i, 'distance'] for i in xs])

            means_distance_ssar = np.mean(distance_ssar, axis=1)
            means_distance_mcs = np.mean(distance_mcs, axis=1)

            std_distance_ssar = np.std(distance_ssar, axis=1)
            std_distance_mcs = np.std(distance_mcs, axis=1)

            f = plt.figure()
            ax = plt.subplot(111)
            ax.plot(xs, means_distance_ssar, label="sSAR {}".format(samples), marker='.')
            # ax.fill_between(xs, means_distance50 - std_distance50, means_distance50 + std_distance50, alpha=0.2)

            ax.plot(xs, means_distance_mcs, label="MCS {}".format(samples), marker='.')
            # ax.fill_between(xs, means_distance10 - std_distance10, means_distance10 + std_distance10, alpha=0.2)

            plt.grid(True, alpha=self.grid_alpha)
            plt.xlabel("Generations", fontsize=self.fontsize)
            plt.ylabel("Hypervolume", fontsize=self.fontsize)
            plt.legend(fontsize=self.fontsize)
            plt.xticks(fontsize=self.ticksize)
            plt.yticks(fontsize=self.ticksize)
            plt.tight_layout()
            plt.show()
            f.savefig(self.plot_folder + "distance_mcs_vs_ssar_{}".format(samples) + ".pdf")


    def _plot_objectives_ssar_vs_mcs(self, column_name='mean'):
        xs = np.arange(0, 51)

        for samples in self.dfs_ssar.keys():
            df_ssar = self.dfs_ssar[samples]
            df_mcs = self.dfs_mcs[samples]

            means_ssar = np.array([np.mean(df_ssar.loc[df_ssar['gen'] == i, column_name]) for i in xs])
            means_mcs = np.array([np.mean(df_mcs.loc[df_mcs['gen'] == i, column_name]) for i in xs])

            title_prefix = "Best" if column_name == 'best' else "Average"

            f = plt.figure()
            ax = plt.subplot(111)
            # plt.title("{} MTTF {} samples".format(title_prefix, samples), fontsize=self.fontsize)
            plt.plot(xs, means_mcs.T[0] / (24 * 365), marker='^', label="MCS {}".format(samples))
            plt.plot(xs, means_ssar.T[0] / (24 * 365), marker='^', label="sSAR {}".format(samples))
            plt.legend(fontsize=self.fontsize)
            plt.grid(True, alpha=self.grid_alpha)
            plt.xticks(fontsize=self.ticksize)
            plt.yticks(fontsize=self.ticksize)
            plt.xlabel("Generations", fontsize=self.fontsize)
            plt.ylabel("MTTF in years", fontsize=self.fontsize)
            ins = ax.inset_axes([0.655, 0.4, 0.3, 0.3])
            ins.plot(xs[-6:], means_mcs.T[0][-6:] / (24 * 365), marker='^')
            ins.plot(xs[-6:], means_ssar.T[0][-6:] / (24 * 365), marker='^')
            ax.indicate_inset_zoom(ins, alpha=0.6, edgecolor='#000000')
            plt.show()
            plt.tight_layout()
            f.savefig(self.plot_folder + "{}_mttf_ssar_vs_mcs_{}".format(column_name, samples) + ".pdf")

            f = plt.figure()
            ax = plt.subplot(111)
            # plt.title("{} energy objective {} samples".format(title_prefix, samples), fontsize=self.fontsize)
            plt.plot(xs, means_mcs.T[1], marker='^', label="MCS {}".format(samples))
            plt.plot(xs, means_ssar.T[1], marker='^', label="sSAR {}".format(samples))
            plt.legend(fontsize=self.fontsize)
            plt.grid(True, alpha=self.grid_alpha)
            plt.xticks(fontsize=self.ticksize)
            plt.yticks(fontsize=self.ticksize)
            plt.xlabel("Generations", fontsize=self.fontsize)
            plt.ylabel("Energy consumption", fontsize=self.fontsize)
            if column_name != 'best':
                ins = ax.inset_axes([0.655, 0.4, 0.3, 0.3])
                ins.plot(xs[-6:], means_mcs.T[1][-6:], marker='^')
                ins.plot(xs[-6:], means_ssar.T[1][-6:], marker='^')
                ax.indicate_inset_zoom(ins, alpha=0.6, edgecolor='#000000')
            plt.show()
            plt.tight_layout()
            f.savefig(self.plot_folder + "{}_pe_ssar_vs_mcs_{}".format(column_name, samples) + ".pdf")

            f = plt.figure()
            ax = plt.subplot(111)
            # plt.title("{} size {} samples".format(title_prefix, samples), fontsize=self.fontsize)
            plt.plot(xs, means_mcs.T[2], marker='^', label="MCS {}".format(samples))
            plt.plot(xs, means_ssar.T[2], marker='^', label="sSAR {}".format(samples))
            plt.legend(fontsize=self.fontsize)
            plt.grid(True, alpha=self.grid_alpha)
            plt.xticks(fontsize=self.ticksize)
            plt.yticks(fontsize=self.ticksize)
            plt.xlabel("Generations", fontsize=self.fontsize)
            plt.ylabel("Size", fontsize=self.fontsize)
            if column_name != 'best':
                ins = ax.inset_axes([0.655, 0.4, 0.3, 0.3])
                ins.plot(xs[-6:], means_mcs.T[2][-6:], marker='^')
                ins.plot(xs[-6:], means_ssar.T[2][-6:], marker='^')
                ax.indicate_inset_zoom(ins, alpha=0.4, edgecolor='#000000')
            plt.show()
            plt.tight_layout()
            f.savefig(self.plot_folder + "{}_size_ssar_vs_mcs_{}".format(column_name, samples) + ".pdf")


    def _plot_mean_mttf_different_samples(self):
        xs = np.arange(0, 51)

        means100 = np.array([np.mean(self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'mean']) for i in range(0, 51)])
        means50 = np.array([np.mean(self.df_mcs50.loc[self.df_mcs50['gen'] == i, 'mean']) for i in range(0, 51)])
        means10 = np.array([np.mean(self.df_mcs10.loc[self.df_mcs10['gen'] == i, 'mean']) for i in range(0, 51)])
        means1 = np.array([np.mean(self.df_mcs1.loc[self.df_mcs1['gen'] == i, 'mean']) for i in range(0, 51)])

        means_mttf100 = means100.T[0] / (24 * 365)
        means_mttf50 = means50.T[0] / (24 * 365)
        means_mttf10 = means10.T[0] / (24 * 365)
        means_mttf1 = means1.T[0] / (24 * 365)

        f = plt.figure()
        ax = plt.subplot(111)
        ax.plot(xs, means_mttf100, label="100 samples")
        ax.plot(xs, means_mttf50, label="50 samples")
        ax.plot(xs, means_mttf10, label="10 samples")
        ax.plot(xs, means_mttf1, label="1 sample")

        plt.grid(True, alpha=self.grid_alpha)

        ins = ax.inset_axes([0.655, 0.4, 0.3, 0.3])
        ins.plot(xs[-6:], means_mttf100[-6:])
        ins.plot(xs[-6:], means_mttf50[-6:])
        ins.plot(xs[-6:], means_mttf10[-6:])
        ins.plot(xs[-6:], means_mttf1[-6:])

        plt.xlabel("Generations", fontsize=self.fontsize)
        plt.ylabel("MTTF in years", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize)
        plt.show()
        f.savefig(self.plot_folder + "mean_mttf_different_samples" + ".pdf")

    def _plot_mean_pe_different_samples(self):
        xs = np.arange(0, 51)

        means100 = np.array([np.mean(self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'mean']) for i in range(0, 51)])
        means50 = np.array([np.mean(self.df_mcs50.loc[self.df_mcs50['gen'] == i, 'mean']) for i in range(0, 51)])
        means10 = np.array([np.mean(self.df_mcs10.loc[self.df_mcs10['gen'] == i, 'mean']) for i in range(0, 51)])
        means1 = np.array([np.mean(self.df_mcs1.loc[self.df_mcs1['gen'] == i, 'mean']) for i in range(0, 51)])

        means_pe100 = means100.T[1]
        means_pe50 = means50.T[1]
        means_pe10 = means10.T[1]
        means_pe1 = means1.T[1]

        f = plt.figure()
        ax = plt.subplot(111)
        ax.plot(xs, means_pe100, label="100 samples")
        ax.plot(xs, means_pe50, label="50 samples")
        ax.plot(xs, means_pe10, label="10 samples")
        ax.plot(xs, means_pe1, label="1 sample")

        ins = ax.inset_axes([0.655, 0.33, 0.3, 0.3])
        ins.plot(xs[-6:], means_pe100[-6:])
        ins.plot(xs[-6:], means_pe50[-6:])
        ins.plot(xs[-6:], means_pe10[-6:])
        ins.plot(xs[-6:], means_pe1[-6:])

        plt.grid(True, alpha=self.grid_alpha)

        plt.xlabel("Generations", fontsize=self.fontsize)
        plt.ylabel("Energy consumption", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize)
        plt.show()
        f.savefig(self.plot_folder + "mean_pe_different_samples" + ".pdf")

    def _plot_mean_size_different_samples(self):
        xs = np.arange(0, 51)

        means100 = np.array([np.mean(self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'mean']) for i in range(0, 51)])
        means50 = np.array([np.mean(self.df_mcs50.loc[self.df_mcs50['gen'] == i, 'mean']) for i in range(0, 51)])
        means10 = np.array([np.mean(self.df_mcs10.loc[self.df_mcs10['gen'] == i, 'mean']) for i in range(0, 51)])
        means1 = np.array([np.mean(self.df_mcs1.loc[self.df_mcs1['gen'] == i, 'mean']) for i in range(0, 51)])

        means_size100 = means100.T[2]
        means_size50 = means50.T[2]
        means_size10 = means10.T[2]
        means_size1 = means1.T[2]

        f = plt.figure()
        ax = plt.subplot(111)
        ax.plot(xs, means_size100, label="100 samples")
        ax.plot(xs, means_size50, label="50 samples")
        ax.plot(xs, means_size10, label="10 samples")
        ax.plot(xs, means_size1, label="1 sample")

        ins = ax.inset_axes([0.655, 0.365, 0.3, 0.3])
        ins.plot(xs[-6:], means_size100[-6:])
        ins.plot(xs[-6:], means_size50[-6:])
        ins.plot(xs[-6:], means_size10[-6:])
        ins.plot(xs[-6:], means_size1[-6:])

        plt.grid(True, alpha=self.grid_alpha)

        plt.xlabel("Generations", fontsize=self.fontsize)
        plt.ylabel("Size", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize)
        plt.show()
        f.savefig(self.plot_folder + "mean_size_different_samples" + ".pdf")

    def _plot_best_mttf_different_samples(self):
        xs = np.arange(0, 51)

        data100 = np.array([self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'best'] for i in range(0, 51)], dtype=float)
        data50 = np.array([self.df_mcs50.loc[self.df_mcs50['gen'] == i, 'best'] for i in range(0, 51)], dtype=float)
        data10 = np.array([self.df_mcs10.loc[self.df_mcs10['gen'] == i, 'best'] for i in range(0, 51)], dtype=float)
        data1 = np.array([self.df_mcs1.loc[self.df_mcs1['gen'] == i, 'best'] for i in range(0, 51)], dtype=float)

        means_mttf100 = np.mean(data100.T[0] / (24 * 365), axis=0)
        means_mttf50 = np.mean(data50.T[0] / (24 * 365), axis=0)
        means_mttf10 = np.mean(data10.T[0] / (24 * 365), axis=0)
        means_mttf1 = np.mean(data1.T[0] / (24 * 365), axis=0)

        f = plt.figure()
        ax = plt.subplot(111)
        ax.plot(xs, means_mttf100, label="100 samples")
        ax.plot(xs, means_mttf50, label="50 samples")
        ax.plot(xs, means_mttf10, label="10 samples")
        ax.plot(xs, means_mttf1, label="1 sample")

        ins = ax.inset_axes([0.655, 0.33, 0.3, 0.25])
        ins.plot(xs[-6:], means_mttf100[-6:])
        ins.plot(xs[-6:], means_mttf50[-6:])
        ins.plot(xs[-6:], means_mttf10[-6:])
        ins.plot(xs[-6:], means_mttf1[-6:])

        plt.grid(True, alpha=self.grid_alpha)
        plt.xlabel("Generations", fontsize=self.fontsize)
        plt.ylabel("MTTF in years", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize, loc='upper left')
        plt.show()
        f.savefig(self.plot_folder + "best_mttf_different_samples" + ".pdf")

    def _plot_best_pe_different_samples(self):
        xs = np.arange(0, 51)

        data100 = np.array([self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'best'] for i in range(0, 51)], dtype=float)
        data50 = np.array([self.df_mcs50.loc[self.df_mcs50['gen'] == i, 'best'] for i in range(0, 51)], dtype=float)
        data10 = np.array([self.df_mcs10.loc[self.df_mcs10['gen'] == i, 'best'] for i in range(0, 51)], dtype=float)
        data1 = np.array([self.df_mcs1.loc[self.df_mcs1['gen'] == i, 'best'] for i in range(0, 51)], dtype=float)

        means_pe100 = np.mean(data100.T[1], axis=0)
        means_pe50 = np.mean(data50.T[1], axis=0)
        means_pe10 = np.mean(data10.T[1], axis=0)
        means_pe1 = np.mean(data1.T[1], axis=0)

        f = plt.figure()
        ax = plt.subplot(111)
        ax.plot(xs, means_pe100, label="100 samples")
        ax.plot(xs, means_pe50, label="50 samples")
        ax.plot(xs, means_pe10, label="10 samples")
        ax.plot(xs, means_pe1, label="1 sample")

        plt.grid(True, alpha=self.grid_alpha)
        plt.xlabel("Generations", fontsize=self.fontsize)
        plt.ylabel("Energy consumption", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize, loc='upper right')
        plt.show()
        f.savefig(self.plot_folder + "best_pe_different_samples" + ".pdf")

    def _plot_best_size_different_samples(self):
        xs = np.arange(0, 51)

        data100 = np.array([self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'best'] for i in range(0, 51)], dtype=float)
        data50 = np.array([self.df_mcs50.loc[self.df_mcs50['gen'] == i, 'best'] for i in range(0, 51)], dtype=float)
        data10 = np.array([self.df_mcs10.loc[self.df_mcs10['gen'] == i, 'best'] for i in range(0, 51)], dtype=float)
        data1 = np.array([self.df_mcs1.loc[self.df_mcs1['gen'] == i, 'best'] for i in range(0, 51)], dtype=float)

        means_size100 = np.mean(data100.T[2], axis=0)
        means_size50 = np.mean(data50.T[2], axis=0)
        means_size10 = np.mean(data10.T[2], axis=0)
        means_size1 = np.mean(data1.T[2], axis=0)

        f = plt.figure()
        ax = plt.subplot(111)
        ax.plot(xs, means_size100, label="100 samples")
        ax.plot(xs, means_size50, label="50 samples")
        ax.plot(xs, means_size10, label="10 samples")
        ax.plot(xs, means_size1, label="1 sample")

        plt.grid(True, alpha=self.grid_alpha)
        plt.xlabel("Generations", fontsize=self.fontsize)
        plt.ylabel("Size", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize, loc='upper right')
        plt.show()
        f.savefig(self.plot_folder + "best_size_different_samples" + ".pdf")

    def _plot_elitism_different_samples(self):
        nr_elitism100 = np.array([self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'elitism'] for i in range(0, 51)])
        nr_elitism50 = np.array([self.df_mcs50.loc[self.df_mcs50['gen'] == i, 'elitism'] for i in range(0, 51)])
        nr_elitism10 = np.array([self.df_mcs10.loc[self.df_mcs10['gen'] == i, 'elitism'] for i in range(0, 51)])
        nr_elitism1 = np.array([self.df_mcs1.loc[self.df_mcs1['gen'] == i, 'elitism'] for i in range(0, 51)])

        means100 = np.mean(nr_elitism100, axis=1) / 100
        means50 = np.mean(nr_elitism50, axis=1) / 100
        means10 = np.mean(nr_elitism10, axis=1) / 100
        means1 = np.mean(nr_elitism1, axis=1) / 100

        std100 = np.std(nr_elitism100, axis=1) / 100
        std50 = np.std(nr_elitism10, axis=1) / 100
        std10 = np.std(nr_elitism10, axis=1) / 100
        std1 = np.std(nr_elitism1, axis=1) / 100

        xs = np.arange(0, 51)
        f = plt.figure()
        ax = plt.subplot(111)

        ax.plot(xs, means100, label="100 samples")
        ax.plot(xs, means50, label="50 samples")
        ax.plot(xs, means10, label="10 samples")
        ax.plot(xs, means1, label="1 sample")

        ins = ax.inset_axes([0.655, 0.4, 0.3, 0.3])
        ins.set_xlim(45, 50)
        ins.set_ylim(0.89, 0.945)
        ins.plot(xs[-6:], means100[-6:])
        ins.plot(xs[-6:], means50[-6:])
        ins.plot(xs[-6:], means10[-6:])
        ins.plot(xs[-6:], means1[-6:])

        plt.grid(True, alpha=self.grid_alpha)
        plt.xlabel("Generations", fontsize=self.fontsize)
        plt.ylabel("Elitism in %", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize, loc='lower right')
        plt.show()
        f.savefig(self.plot_folder + "elitism_different_samples" + ".pdf")

    def _plot_average_elitism_over_gens(self):
        nr_elitism = np.array([self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'elitism'] for i in range(0, 51)])

        means = np.mean(nr_elitism, axis=1) / 100
        std = np.std(nr_elitism, axis=1) / 100

        xs = np.arange(0, 51)

        f = plt.figure()
        ax = plt.subplot(111)
        ax.plot(xs, means, color='b', label="Mean")
        ax.fill_between(xs, means - std, means + std,
                        color='b', alpha=0.2, label="SD")

        plt.grid(True, alpha=self.grid_alpha)
        plt.xlabel("Generations", fontsize=self.fontsize)
        plt.ylabel("Elitism in %", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize, loc='lower right')
        plt.show()
        f.savefig(self.plot_folder + "average_elitism_over_gens" + ".pdf")

    def _plot_offspring_and_mutations_over_gens(self):
        nr_offspring = np.array([self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'offspring'] for i in range(0, 51)])
        nr_mutations = np.array([self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'mutations'] for i in range(0, 51)])
        death_penalty = np.array([self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'death_penalty'] for i in range(0, 51)])

        means_offspring = np.mean(nr_offspring, axis=1)
        means_mutations = np.mean(nr_mutations, axis=1)
        means_death_penalty = np.mean(death_penalty, axis=1)

        std_offpsring = np.std(nr_offspring, axis=1)
        std_mutations = np.std(nr_mutations, axis=1)
        std_death_penalty = np.std(death_penalty, axis=1)

        xs = np.arange(0, 51)

        f = plt.figure()
        ax = plt.subplot(111)
        ax.plot(xs, means_offspring, label="Offspring")
        ax.fill_between(xs, means_offspring - std_offpsring, means_offspring + std_offpsring, alpha=0.2)

        ax.plot(xs, means_mutations, label="Mutations")
        ax.fill_between(xs, means_mutations - std_mutations, means_mutations + std_mutations, alpha=0.2)

        ax.plot(xs, means_death_penalty, label="Invalids")
        ax.fill_between(xs, means_death_penalty - std_death_penalty, means_death_penalty + std_death_penalty,
                        alpha=0.2)
        plt.grid(True, alpha=self.grid_alpha)
        plt.xlabel("Generations", fontsize=self.fontsize)
        plt.ylabel("Average occurrences", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize)
        plt.show()
        f.savefig(self.plot_folder + "offspring_and_mutations_over_gens" + ".pdf")

    def _plot_distance_over_gens(self):
        distance50 = np.array([self.df_mcs50.loc[self.df_mcs50['gen'] == i, 'distance'] for i in range(0, 51)])
        distance10 = np.array([self.df_mcs10.loc[self.df_mcs10['gen'] == i, 'distance'] for i in range(0, 51)])
        distance1 = np.array([self.df_mcs1.loc[self.df_mcs1['gen'] == i, 'distance'] for i in range(0, 51)])

        means_distance50 = np.mean(distance50, axis=1)
        means_distance10 = np.mean(distance10, axis=1)
        means_distance1 = np.mean(distance1, axis=1)

        std_distance50 = np.std(distance50, axis=1)
        std_distance10 = np.std(distance10, axis=1)
        std_distance1 = np.std(distance1, axis=1)

        xs = np.arange(0, 51)

        f = plt.figure()
        ax = plt.subplot(111)
        ax.plot(xs, means_distance50, label="50 samples")
        # ax.fill_between(xs, means_distance50 - std_distance50, means_distance50 + std_distance50, alpha=0.2)

        ax.plot(xs, means_distance10, label="10 samples")
        # ax.fill_between(xs, means_distance10 - std_distance10, means_distance10 + std_distance10, alpha=0.2)

        ax.plot(xs, means_distance1, label="1 sample")
        # ax.fill_between(xs, means_distance1 - std_distance1, means_distance1 + std_distance1, alpha=0.2)

        plt.grid(True, alpha=self.grid_alpha)
        plt.xlabel("Generations", fontsize=self.fontsize)
        plt.ylabel("Hypervolume", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize)
        plt.show()
        f.savefig(self.plot_folder + "distance_over_gens" + ".pdf")

    def _plot_mean_mttf_over_gens(self):
        xs = np.arange(0, 51)

        means = np.array([np.mean(self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'mean']) for i in range(0, 51)])
        means_mttf = means.T[0] / (24 * 365)

        std = np.array([np.mean(self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'std']) for i in range(0, 51)])
        std_mttf = std.T[0] / (24 * 365)

        f = plt.figure()
        ax = plt.subplot(111)
        ax.plot(xs, means_mttf, color='c', label="Mean")
        ax.fill_between(xs, means_mttf - std_mttf, means_mttf + std_mttf,
                        color='c', alpha=0.2, label="SD")
        plt.grid(True, alpha=self.grid_alpha)

        plt.xlabel("Generations", fontsize=self.fontsize)
        plt.ylabel("MTTF in years", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize, loc='lower right')
        plt.show()
        f.savefig(self.plot_folder + "mean_mttf_over_gens" + ".pdf")

    def _plot_mean_pe_over_gens(self):
        xs = np.arange(0, 51)

        means = np.array([np.mean(self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'mean']) for i in range(0, 51)])
        means_pe = means.T[1]

        std = np.array([np.mean(self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'std']) for i in range(0, 51)])
        std_pe = std.T[1]

        f = plt.figure()
        ax = plt.subplot(111)
        ax.plot(xs, means_pe, color='y', label="Mean")
        ax.fill_between(xs, means_pe - std_pe, means_pe + std_pe,
                        color='y', alpha=0.2, label="SD")
        plt.grid(True, alpha=self.grid_alpha)

        plt.xlabel("Generations", fontsize=self.fontsize)
        plt.ylabel("Energy consumption", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize, loc='upper right')
        plt.show()
        f.savefig(self.plot_folder + "mean_pe_over_gens" + ".pdf")

    def _plot_mean_size_over_gens(self):
        xs = np.arange(0, 51)

        means = np.array([np.mean(self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'mean']) for i in range(0, 51)])
        means_size = means.T[2]

        std = np.array([np.mean(self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'std']) for i in range(0, 51)])
        std_size = std.T[2]

        f = plt.figure()
        ax = plt.subplot(111)
        ax.plot(xs, means_size, color='g', label="Mean")
        ax.fill_between(xs, means_size - std_size, means_size + std_size,
                        color='g', alpha=0.2, label="SD")
        plt.grid(True, alpha=self.grid_alpha)
        plt.xlabel("Generations", fontsize=self.fontsize)
        plt.ylabel("Size", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize, loc='upper right')
        plt.show()
        f.savefig(self.plot_folder + "mean_size_over_gens" + ".pdf")

    def _plot_best_mttf_over_gens(self):
        xs = np.arange(0, 51)

        data = np.array([self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'best'] for i in range(0, 51)], dtype=float)

        means_mttf = np.mean(data.T[0] / (24 * 365), axis=0)
        std_mttf = np.std(data.T[0] / (24 * 365), axis=0)

        f = plt.figure()
        ax = plt.subplot(111)
        ax.plot(xs, means_mttf, color='c', label="Mean")
        ax.fill_between(xs, means_mttf - std_mttf, means_mttf + std_mttf, color='c', label="SD",
                        alpha=0.2)
        plt.grid(True, alpha=self.grid_alpha)
        plt.xlabel("Generations", fontsize=self.fontsize)
        plt.ylabel("MTTF in years", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize, loc='lower right')
        plt.show()
        f.savefig(self.plot_folder + "best_mttf_over_gens" + ".pdf")

    def _plot_best_pe_over_gens(self):
        xs = np.arange(0, 51)

        data = np.array([self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'best'] for i in range(0, 51)], dtype=float)
        means_pe = np.mean(data.T[1], axis=0)
        std_pe = np.std(data.T[1], axis=0)

        f = plt.figure()
        ax = plt.subplot(111)
        ax.plot(xs, means_pe, color='y', label="Mean")
        ax.fill_between(xs, means_pe - std_pe, means_pe + std_pe, color='y',
                        label="SD", alpha=0.2)
        plt.grid(True, alpha=self.grid_alpha)
        plt.xlabel("Generations", fontsize=self.fontsize)
        plt.ylabel("Energy consumption", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize, loc='upper right')
        plt.show()
        f.savefig(self.plot_folder + "best_pe_over_gens" + ".pdf")

    def _plot_best_size_over_gens(self):
        xs = np.arange(0, 51)
        
        data = np.array([self.df_mcs100.loc[self.df_mcs100['gen'] == i, 'best'] for i in range(0, 51)], dtype=float)

        means_size = np.mean(data.T[2], axis=0)
        std_size = np.std(data.T[2], axis=0)

        f = plt.figure()
        ax = plt.subplot(111)
        ax.plot(xs, means_size, color='g', label="Mean")
        ax.fill_between(xs, means_size - std_size, means_size + std_size, color='g', label="SD",
                        alpha=0.2)
        plt.grid(True, alpha=self.grid_alpha)
        plt.xlabel("Generations", fontsize=self.fontsize)
        plt.ylabel("Size", fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize, loc='upper right')
        plt.show()
        f.savefig(self.plot_folder + "best_size_over_gens" + ".pdf")

    def plot_pareto_front_mttf_pe(self):
        front_values = self.analysis.pareto_front(use_objectives=[1.0, 1.0, 0.0])
        front = np.array([i.fitness.values for i in front_values])
        data = np.array(list(self.analysis.data.values()))

        mean_data = np.asarray(list(self.analysis2.means().values()))

        f = plt.figure()
        plt.scatter(mean_data.T[1], mean_data.T[0] / (24 * 365), marker='.', label='Random')
        plt.scatter(data.T[1], data.T[0] / (24 * 365), marker='^', c="#bc6c25", label='GA output')
        plt.scatter(front.T[1], front.T[0] / (24 * 365), marker='o', c='#9d0208', label="Pareto optimal")

        plt.xlabel("Energy consumption", fontsize=self.fontsize)
        plt.ylabel("MTTF in years", fontsize=self.fontsize)

        plt.legend(fontsize=self.fontsize)
        plt.show()
        f.savefig("out/plots/simulator/plot_pareto_front_mttf.pdf", bbox_inches='tight')

    def plot_pareto_front_mttf_size(self):
        front_values = self.analysis.pareto_front(use_objectives=[1.0, 0.0, 1.0])
        front = np.array([i.fitness.values for i in front_values])
        data = np.array(list(self.analysis.data.values()))

        mean_data = np.asarray(list(self.analysis2.means().values()))

        f = plt.figure()
        plt.scatter(mean_data.T[2], mean_data.T[0] / (24 * 365), marker='.', label='Random')
        plt.scatter(data.T[2], data.T[0] / (24 * 365), marker='^', c="#bc6c25", label='GA output')
        plt.scatter(front.T[2], front.T[0] / (24 * 365), marker='o', c='#9d0208', label="Pareto optimal")

        plt.xlabel("Size", fontsize=self.fontsize)
        plt.ylabel("MTTF in years", fontsize=self.fontsize)

        # plt.legend(fontsize=self.fontsize, loc=5)
        plt.show()
        f.savefig("out/plots/simulator/plot_pareto_front_mttf_size.pdf", bbox_inches='tight')

    def plot_pareto_front_pe_size(self):
        front_values = self.analysis.pareto_front(use_objectives=[0.0, 1.0, 1.0])
        front = np.array([i.fitness.values for i in front_values])
        data = np.array(list(self.analysis.data.values()))

        mean_data = np.asarray(list(self.analysis2.means().values()))

        f = plt.figure()
        plt.scatter(mean_data.T[1], mean_data.T[2], marker='.', label='Random')
        plt.scatter(data.T[1], data.T[2], marker='^', c="#bc6c25", label='GA output')
        plt.scatter(front.T[1], front.T[2], marker='o', c='#9d0208', label="Pareto optimal")

        plt.xlabel("Energy consumption", fontsize=self.fontsize)
        plt.ylabel("Size", fontsize=self.fontsize)

        plt.legend(fontsize=self.fontsize)
        plt.show()
        f.savefig("out/plots/simulator/plot_pareto_front_pe_size.pdf", bbox_inches='tight')


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


def obtain_all_plots():
    # plottingMCS = PlottingMCS()    #
    plottingGA = PlottingGA()

    print("Starting plotting functions")

    # plottingMCS.objective_space_plots()
    plottingGA.plot_pareto_front_mttf_pe()
    plottingGA.plot_pareto_front_mttf_size()
    plottingGA.plot_pareto_front_pe_size()


if __name__ == "__main__":
    plotter = PlottingEvaluation()


    # mcs1 = {5: 135, 10: 97, 15: 79, 20: 61, 25: 70, 30: 69, 35: 57, 40: 44, 45: 45, 50: 39, 55: 56, 60: 55, 65: 39, 70: 41, 75: 37, 80: 45, 85: 35, 90: 43, 95: 36, 100: 37, 105: 30, 110: 28, 115: 30, 120: 31, 125: 32, 130: 24, 135: 35, 140: 27, 145: 30, 150: 32, 155: 25, 160: 25, 165: 13, 170: 31, 175: 26, 180: 28, 185: 19, 190: 25, 195: 24, 200: 23, 205: 30}
    # mcs2 = {5: 142, 10: 108, 15: 87, 20: 71, 25: 60, 30: 75, 35: 54, 40: 48, 45: 56, 50: 56, 55: 46, 60: 46, 65: 40, 70: 34, 75: 37, 80: 41, 85: 32, 90: 27, 95: 31, 100: 26, 105: 33, 110: 29, 115: 30, 120: 23, 125: 29, 130: 25, 135: 30, 140: 24, 145: 29, 150: 36, 155: 18, 160: 25, 165: 23, 170: 27, 175: 22, 180: 23, 185: 23, 190: 24, 195: 17, 200: 26, 205: 27}
    #
    # sSAR1 = {(5, 493): 117, (10, 1066): 106, (15, 1632): 89, (20, 2195): 82, (25, 2803): 70, (30, 3365): 74, (35, 3932): 61, (40, 4499): 63, (45, 5090): 52, (50, 5663): 52, (55, 6230): 56, (60, 6820): 56, (65, 7391): 43, (70, 7954): 63, (75, 8519): 43, (80, 9098): 40, (85, 9694): 38, (90, 10246): 35, (95, 10831): 51, (100, 11397): 42, (105, 11988): 32, (110, 12559): 40, (115, 13133): 33, (120, 13701): 35, (125, 14296): 32, (130, 14857): 33, (135, 15448): 36, (140, 15986): 37, (145, 16605): 45, (150, 17131): 32, (155, 17696): 34,  (160, 18308): 21, (165, 18893): 37, (170, 19451): 21, (175, 20027): 32, (180, 20625): 38, (185, 21153): 35, (190, 21762): 28, (195, 22320): 32, (200, 22901): 32, (205, 23489): 31}
    #
    # pUCB1 = {5: 138, 10: 100, 15: 84, 20: 65, 25: 71, 30: 70, 35: 62, 40: 50, 45: 53, 50: 51, 55: 43, 60: 48, 65: 35, 70: 41, 75: 37, 80: 37, 85: 40, 90: 39, 95: 45, 100: 38, 105: 29, 110: 40, 115: 27, 120: 28, 125: 23, 130: 32, 135: 38, 140: 28, 145: 32, 150: 33, 155: 22, 160: 25, 165: 26, 170: 30, 175: 24, 180: 31, 185: 30, 190: 35, 195: 33, 200: 26, 205: 31}
    # pUCB2 = {5: 143, 10: 100, 15: 96, 20: 88, 25: 67, 30: 70, 35: 51, 40: 53, 45: 56, 50: 51, 55: 43, 60: 42, 65: 51, 70: 34, 75: 35, 80: 42, 85: 41, 90: 33, 95: 35, 100: 31, 105: 41, 110: 41, 115: 34, 120: 34, 125: 32, 130: 37, 135: 35, 140: 26, 145: 32, 150: 28, 155: 24, 160: 28, 165: 34, 170: 24, 175: 32, 180: 30, 185: 30, 190: 24, 195: 29, 200: 23, 205: 21}
    #
    # f = plt.figure()
    # plt.plot(list(mcs1.keys()), list(mcs1.values()), label="mcs")
    # plt.plot([k[1] // 50 for k in sSAR1.keys()], list(sSAR1.values()), label="sSAR")
    #
    # plt.plot(list(pUCB1.keys()), list(pUCB1.values()), label="pucb")
    #
    # plt.xlabel("Samples per individual", fontsize=14)
    # plt.ylabel("Incorrect selections", fontsize=14)
    #
    # plt.legend(fontsize=14)
    # plt.show()
    # f.savefig("out/plots/test_wrong_selected.pdf")