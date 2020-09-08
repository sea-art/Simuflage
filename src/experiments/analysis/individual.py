import numpy as np

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

    @property
    def genes(self):
        return self.individual.genes

    def __repr__(self):
        return str(self.individual)

    def mcs(self, nr_samples, average=True):
        samples = self.samples[np.random.choice(self.samples.shape[0], nr_samples, replace=False)]

        return tuple(np.mean(samples, axis=0)) if average else samples