#!/usr/bin/env python

""" Contains methods to explore the design space exploration with genetic algrorithms.

TODO:
maybe its possible to separate all the inputs for a design point and create GA methods/operators for each individual
part. E.g.:
    - application mapping can be a "sample" of the predefined list of possible application mappings maps with constraints (with Tool Decoration)
    - capacities can be random selection of predefined list
    - position can also be random selection of predefined list of all possible locations
    - amount of components (maybe mutation might be to either add or remove 1 component ?)

Functions can also call each other, so the latest "individual" function, can just call all other functions and create dp with data


"""

import random
from copy import deepcopy, copy

from deap import base
from deap import creator
from deap import tools

from DSE import monte_carlo
from design import DesignPoint, Component, Application
from design.mapping import all_possible_pos_mappings


loc_choices = list(map(tuple, all_possible_pos_mappings(26)))

maps = [(0, 0), (1, 1), (2, 2), (3, 3)]
capacity_candidates = [44, 88, 100, 150, 200, 600]
apps = [10, 10, 10, 10]
CXPB, MUTPB = 0.5, 0.3


def init_dp(pcls, locs=None, caps=None, init_apps=None, policy=None):
    if not caps:
        caps = random.choices(capacity_candidates, k=4)

    if not locs:
        locs = random.sample(loc_choices, 4)

    if not policy:
        policy = random.choice(['most', 'least', 'random'])

    caps = [Component(caps[i], locs[i]) for i in range(len(caps))]
    zapps = [Application(a) for a in apps]
    app_map = [(caps[i], zapps[i]) for i in range(len(caps))]

    return pcls(caps, zapps, app_map, policy)


def mate_dps(x1, x2):
    caps1 = [z._capacity for z in x1._components[:2] + x2._components[2:]]
    caps2 = [z._capacity for z in x2._components[:2] + x1._components[2:]]

    locs1 = [z._loc for z in x1._components]
    locs2 = [z._loc for z in x2._components]

    policy1 = x1.policy
    policy2 = x2.policy

    x1 = toolbox.designpoint(caps=caps1, locs=locs1, policy=policy1)
    x2 = toolbox.designpoint(caps=caps2, locs=locs2, policy=policy2)

    return x1, x2


def mutate_dp(x1):
    caps = [z._capacity for z in x1._components]
    locs = [z._loc for z in x1._components]

    policy = x1.policy
    possible_policies = ['most', 'least', 'random']
    possible_policies.remove(policy)

    caps_index = random.randint(0, 3)
    loc_index = random.randint(0, 3)

    possible_candidates = copy(capacity_candidates)
    possible_candidates.remove(caps[caps_index])

    loc_candidates = copy(loc_choices)

    for z in range(len(locs)):
        loc_candidates.remove(locs[z])

    tmp_c = random.choice(possible_candidates)
    tmp_l = random.choice(loc_candidates)
    # print("Changing (%s : %s) --[to]--> (%s, %s)" % (caps[i], locs[i], tmp_c, tmp_l))

    caps[caps_index] = tmp_c
    locs[loc_index] = tmp_l

    change_policy = random.randint(0, 2)

    if change_policy == 0:
        policy = random.choice(possible_policies)

    return toolbox.designpoint(caps=caps, locs=locs, policy=policy)


def eval_dp(dp):
    if len(dp) < 1:
        return []

    return list(monte_carlo(dp, iterations=1000, parallelized=True).values())


def mate_offspring(offspring):
    offspring_mated = copy(offspring)

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            x1, x2 = toolbox.mate(child1, child2)
            offspring_mated.remove(child1)
            offspring_mated.remove(child2)
            offspring_mated.append(x1)
            offspring_mated.append(x2)

    return offspring_mated


def mutate_offpsring(offspring):
    offspring_mutated = copy(offspring)

    for idx in range(len(offspring)):
        if random.random() < MUTPB:
            x1 = toolbox.mutate(offspring[idx])
            # print("MUTATING", offspring[idx], x1)
            offspring_mutated.remove(offspring[idx])
            offspring_mutated.append(x1)

    return offspring_mutated


def offspring_determination(pop):
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))
    offspring = mate_offspring(offspring)
    offspring = mutate_offpsring(offspring)

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

    fitnesses_mttf = toolbox.evaluate(invalid_ind)
    fitnesses_size = []

    for i in range(len(invalid_ind)):
        fitnesses_size.append(invalid_ind[i].calc_grid_size())

    fitnesses = zip(fitnesses_mttf, fitnesses_size)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(invalid_ind))

    return offspring


def print_status(pop):
    fits = [ind.fitness.values[0] for ind in pop]
    fits_size = [ind.fitness.values[1] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    mean_size = sum(fits_size) / length

    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    sum3 = sum(x * x for x in fits_size)
    std2 = abs(sum3 / length - mean_size ** 2) ** 0.5

    print(" \t\t | MTTF \t| Size")
    print("--------------------------------")
    print("  Min \t| %s \t| %s" % (int(min(fits)), min(fits_size)))
    print("  Max \t| %s \t| %s" % (int(max(fits)), max(fits_size)))
    print("  Avg \t| %s \t| %s" % (int(mean), mean_size))
    print("  Std \t| %s \t| %s\n" % (int(std), std2))


creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("DesignPoint", DesignPoint, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register('designpoint', init_dp, creator.DesignPoint)
toolbox.register("population", tools.initRepeat, list, toolbox.designpoint)

toolbox.register("evaluate", eval_dp)
toolbox.register("mate", mate_dps)
toolbox.register("mutate", mutate_dp)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    print("Start of evolution")
    pop = toolbox.population(n=40)

    fitnesses_mttf = toolbox.evaluate(pop)
    fitnesses_size = []

    for i in range(len(pop)):
        fitnesses_size.append(pop[i].calc_grid_size())

    fitnesses = zip(fitnesses_mttf, fitnesses_size)

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))
    g = 0

    while g < 30:
        g = g + 1
        print("Generation %i" % g)

        offspring = offspring_determination(pop)
        pop[:] = offspring
        print_status(pop)

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s % s %s" % (best_ind,
                                                best_ind.fitness.values,
                                                toolbox.evaluate([best_ind]),
                                                best_ind.calc_grid_size()))

    print("-- End of (successful) evolution --")


if __name__ == "__main__":
    main()
