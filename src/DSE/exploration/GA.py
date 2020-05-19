#!/usr/bin/env python

""" Contains methods to explore the design space exploration with genetic algrorithms."""

import random
from copy import deepcopy

from deap import base
from deap import creator
from deap import tools

from DSE import monte_carlo
from design import DesignPoint, Component, Application

locs = [(0, 0), (0, 2), (2, 0), (2, 2)]
maps = [(0, 0), (1, 1), (2, 2), (3, 3)]
capacity_candidates = [44, 88, 100, 200, 300, 600]
apps = [10, 10, 10, 10]


def init_dp(pcls, caps=None, init_apps=None, policy=None):
    if not caps:
        caps = capacity_candidates
        caps = random.choices(caps, k=4)

    if not policy:
        policy = random.choice(['most', 'least', 'random'])

    caps = [Component(caps[i], locs[i]) for i in range(len(caps))]
    zapps = [Application(a) for a in apps]
    app_map = [(caps[i], zapps[i]) for i in range(len(caps))]

    return pcls(caps, zapps, app_map, policy)


def mate_dps(x1, x2):
    caps1 = [z._capacity for z in  x1._components[:2] + x2._components[2:]]
    caps2 = [z._capacity for z in x2._components[:2] + x1._components[2:]]

    policy1 = x1.policy
    policy2 = x2.policy

    x1 = toolbox.designpoint(caps=caps1, policy=policy1)
    x2 = toolbox.designpoint(caps=caps2, policy=policy2)

    return x1, x2


def mutate_dp(x1):
    caps = [z._capacity for z in  x1._components]
    policy = x1.policy

    i = random.randint(0, 3)

    possible_candidates = deepcopy(capacity_candidates)
    possible_candidates.remove(caps[i])

    caps[i] = random.choice(possible_candidates)

    x1 = toolbox.designpoint(caps=caps, policy=policy)
    return x1


def eval_dp(dp):
    if len(dp) < 1:
        return []
    return list(monte_carlo(dp, iterations=1000, parallelized=True).values())


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("DesignPoint", DesignPoint, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('designpoint', init_dp, creator.DesignPoint)
toolbox.register("population", tools.initRepeat, list, toolbox.designpoint)

toolbox.register("evaluate", eval_dp)
toolbox.register("mate", mate_dps)
toolbox.register("mutate", mutate_dp)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    print("Start of evolution")

    pop = toolbox.population(n=20)
    CXPB, MUTPB = 0.5, 0.2

    fitnesses = list(map(lambda x: (x,), toolbox.evaluate(pop)))

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))
    g = 0

    while g < 100:
        g = g + 1
        print("Generation %i" % g)

        offspring = toolbox.select(pop, len(pop))

        offspring = list(map(toolbox.clone, offspring))

        offspring_mated = offspring

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                # print("MATING", child1, child2)
                x1, x2 = toolbox.mate(child1, child2)
                offspring_mated.remove(child1)
                offspring_mated.remove(child2)
                offspring_mated.append(x1)
                offspring_mated.append(x2)

        offspring = offspring_mated

        offspring_mutated = offspring

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            x1 = toolbox.mutate(mutant)
            offspring_mutated.remove(mutant)
            offspring_mutated.append(x1)

        offspring = offspring_mutated

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(lambda x: (x,), toolbox.evaluate(invalid_ind)))

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


if __name__ == "__main__":
    main()
