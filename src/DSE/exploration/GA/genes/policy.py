import numpy as np

class Policy:
    def __init__(self, policy):
        """

        :param policy: TODO: either integer representing index of list of possible policies or string that represents it
        """
        self.policy = policy

    def __repr__(self):
        return self.policy

    def mutate(self, search_space):
        self.policy = np.random.choice(search_space.policies[search_space.policies != self.policy])

    @staticmethod
    def mate(parent1, parent2):
        return Policy(parent1.policy), Policy(parent2.policy)
