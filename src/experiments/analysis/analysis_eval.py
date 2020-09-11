import itertools
import multiprocessing
import pickle
from collections import OrderedDict

import numpy as np
from deap.tools import sortNondominated, selBest, selNSGA2

import experiments.analysis.statics as glbl
from experiments import AnalysisMCS
from experiments.analysis import Dps, shuffle
from experiments.analysis import Individual


def number_of_incorrect_selections(output, identifier, dps, samples_per_dp, eval_method='mcs'):
    """ EXPERIMENT
    Finds the number of incorrect selections that eval_method makes on the given design points
    with the specified samples.

    :param output:
    :param identifier:
    :param dps:
    :param samples_per_dp:
    :param eval_method:
    :return:
    """
    temp = []

    for z, dp in enumerate(dps):
        print(z, "/", len(dps), "done")
        dp.to_sel = len(dp.individuals) // 2
        faults, N = dp.wrong_selected(samples_per_dp * len(dp.individuals), eval_method, number=True)

        if eval_method == 'ssar':
            temp.append((N, faults))
        else:
            temp.append(faults)

    # print(temp)

    if eval_method == 'ssar':
        spent_samples = [x[0] for x in temp]
        incorrect_sel = [x[1] for x in temp]
        output[np.mean(spent_samples) // 100] = sum(incorrect_sel)
    else:
        output[samples_per_dp] = sum(temp)


def distance_of_selections(output, identifier, dps, samples_per_dp, eval_method='mcs'):
    """ EXPERIMENT
    Finds the distance between the selected design points with the given eval_method and
    what should have been selected from the given design points.

    :param output:
    :param identifier:
    :param dps:
    :param samples_per_dp:
    :param eval_method:
    :return:
    """
    temp = []

    for z, dp in enumerate(dps):
        print(z, "/", len(dps), "done")
        dp.to_sel = len(dp.individuals) // 2
        distance, N = dp.distance_selections(samples_per_dp * len(dp.individuals), eval_method, metric='hypervolume')

        if eval_method == 'ssar':
            temp.append((N, distance))
        else:
            temp.append(distance)

    if eval_method == 'ssar':
        spent_samples = [x[0] for x in temp]
        incorrect_sel = [x[1] for x in temp]
        output[np.mean(spent_samples) // 100] = np.mean(incorrect_sel) / 10000000
    else:
        output[samples_per_dp] = np.mean(temp) / 10000000


def start_incorrect_selections_experiment(eval_method='mcs'):
    """ EXPERIMENT

    :param distance: True = uses distance, False = uses nr of incorrect selections
    :return:
    """
    dataset = create_dps_set(sample_dataset=False)[::14]
    # dataset = pickle.load(open("out/pickles/working_dps.p", "rb"))[::8]

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    if eval_method == 'ssar':
        # to_samples = list(range(5, 156, 5))
        to_samples = list(range(5, 156, 80))

    else:
        # to_samples = list(range(5, 656, 20))
        to_samples = list(range(5, 656, 400))

    jobs = []

    for avg in range(1):
        for s in to_samples:
            jobs.append(multiprocessing.Process(target=number_of_incorrect_selections,
                                                args=(return_dict, avg, dataset, s, eval_method)))

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

    return OrderedDict(sorted(return_dict.items()))


def verify_ssar_one_objective():
    """ VELIDATION EXPERIMENT
    Since sSAR has better results than MCS, this tests verifies that this is still the case
    when 2 our of the 3 objectives are fixed.

    :return:
    """
    # dataset = create_dps_set(fix_two_objectives=True)
    dataset = pickle.load(open("out/pickles/working_dps.p", "rb"))[::8]

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    to_samples = list(range(15, 200, 50))

    for avg in range(1):
        for s in to_samples:
            jobs.append(multiprocessing.Process(target=number_of_incorrect_selections,
                                                args=(return_dict, avg, dataset, s, 'ssar')))

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

    print("result", OrderedDict(sorted(return_dict.items())))


def verify_scalarization_detection():
    # dataset = create_dps_set(fix_two_objectives=False)
    dataset = pickle.load(open("out/pickles/working_dps.p", "rb"))

    for dps in dataset:
        selected = set()

        for f in glbl.S:
            fitness_vals = []

            for i, dp in enumerate(dps.dps):
                fitness_vals.append((i, f(dps.normalize(dp.fitness.values))))

            fitness_vals = sorted(fitness_vals, key=lambda x: x[1], reverse=True)

            selection = fitness_vals[:50]
            selection_idcs = [x[0] for x in selection]

            selected = selected.union(set(selection_idcs))

            # incorrect = [i for i in selected_idcs if i not in ref_idcx]


            # print("selected", selected)

        print("|S|: {}".format(len(glbl.S)))
        print("Length of selected set\t\t\t {}".format(len(selected)))
        print("Length of reference set\t\t\t {}".format(len(dps.ref_sel_idcs)))
        print("-" * 40)
        print("Number of correct selections\t {}".format(len([i for i in selected if i in dps.ref_sel_idcs])))
        print("Number of incorrect selections\t {}".format(len([i for i in selected if i not in dps.ref_sel_idcs])))
        print("Number of points not found\t\t {}".format(len([i for i in dps.ref_sel_idcs if i not in selected])))
        print("")


def create_dps_set(fix_two_objectives=False, sample_dataset=False):
    """
    Usefull to obtain dictionary of 700 sets of 100 individuals with each 10.000 samples.
    :return:
    """
    if sample_dataset:
        dps_names = np.array(['dps5.p'])
        samples_names = np.array(['samples5.p'])
    else:
        dps_names = np.array(["dps2.p", "dps3.p", "dps4.p"])
        samples_names = np.array(["samples2.p", "samples3.p", "samples4.p"])

    analysis = AnalysisMCS(dps_names, samples_names)

    pop_size = 100

    keys = list(analysis.data.keys())
    values = list(analysis.data.values())

    if fix_two_objectives:
        for a in values:
            a[:, 1] = 200
            a[:, 2] = 5

    print("Storing", len(keys), "in", len(keys) // pop_size, "chunks")

    dps = [Dps(keys[i:i + pop_size], values[i:i + pop_size]) for i in
           range(0, len(analysis.data.keys()) - pop_size + 1, pop_size)]

    # print("Pickling data")
    # pickle.dump(dps, open("out/pickles/working_dps.p", "wb"))

    return dps


if __name__ == "__main__":
    # verify_scalarization_detection()

    mcs_res = start_incorrect_selections_experiment(eval_method='mcs')
    print(mcs_res)

    print("Pickling data")
    pickle.dump(mcs_res, open("out/pickles/mcs_nr_incorrect_sel.p", "wb"))

    ssar_res = start_incorrect_selections_experiment(eval_method='ssar')

    print("Pickling data")
    pickle.dump(ssar_res, open("out/pickles/ssar_nr_incorrect_sel.p", "wb"))

    print("mcs:", mcs_res)
    print("ssar:", ssar_res)

