"""
Contains global static variables and functions.
"""


import itertools
from random import shuffle

from deap import creator, base

from DSE.exploration.GA import Chromosome

weights = (1.0, -1.0, -1.0)

creator.create("FitnessDSE_mcs", base.Fitness, weights=weights)
creator.create("Individual_mcs", Chromosome, fitness=creator.FitnessDSE_mcs)


def linear_scalarize(vector, w):
    return sum([vector[i] * w[i] for i in range(len(vector))])


def scalarized_lambda(w):
    return lambda vec: linear_scalarize(vec, w)


def get_all_weights(steps=10):
    """ Will create a list of all possible weight values for the scalarization functions.

    :type steps: int representing the stepsize (as 1 / steps)
    :return:
    """
    boxes = 3
    values = steps

    rng = list(range(values + 1)) * boxes
    ans = sorted(set(i for i in itertools.permutations(rng, boxes) if sum(i) == values))

    return list(map(lambda tup: (tup[0]/steps, -tup[1]/steps, -tup[2]/steps), ans))
    # print(list(map(lambda tup: (tup[0]/steps, -tup[1]/steps, -tup[2]/steps), ans)))
    # return [(0, 1, 1)]

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

def update_emperical_mean_set(new_avg_sample, ui, N_ui, N_new):
    n_a = N_ui
    n_b = N_new
    n_ab = n_a + n_b

    for i in range(3):
        x_a = ui[i]
        x_b = new_avg_sample[i]

        delta = x_b - x_a

        ui[i] = x_a + delta * (n_b / n_ab)

    return ui

S = [scalarized_lambda(w) for w in get_all_weights() if w[0] != 0.0][::4]

shuffle(S)
