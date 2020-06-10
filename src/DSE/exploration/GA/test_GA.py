from DSE.exploration.GA import SearchSpace, Chromosome
from DSE.exploration.GA.genes import Components

import numpy as np


if __name__ == "__main__":
    capacities = np.array([10, 25, 55])
    applications = [5, 5, 10, 10]
    max_components = 5
    policies = np.array(['most', 'least', 'random'])

    sesp = SearchSpace(capacities, applications, max_components, policies)

    c1 = Chromosome.create_random(sesp)
    c2 = Chromosome.create_random(sesp)

    print(c1)
    print(c2)

    Chromosome.mate(c1, c2)

    # c = Chromosome.create_random(sesp)
    # print("before\n" + str(c))
    # c.mutate()
    # print("after", c)

