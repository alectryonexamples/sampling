import numpy as np
import copy
import scipy.special
from distributions import Gaussian, MixtureOfGaussians

class MetropolisSampling(object):
    """
    @brief      Metropolis Sampling Algorithm
    """
    def __init__(self, dim, distribution, proposal_distribution):
        """
        @brief      creates sampler

        @param      dim                     dimension of probability space
        @param      distribution            distribution of type ContinuousDistributionBase
        @param      proposal_distribution   object with sample(x) function that will generate (symmetric) samples around x
        """ 
        self.dim = dim
        self.distribution = distribution
        self.proposal_distribution = proposal_distribution
        self.x = None
        self.init_x()

    def init_x(self, x=None):
        if x is None:
            self.x = np.random.sample(self.dim)
        else:
            if len(x) != self.dim:
                raise Exception("x is not right size.")
            self.x = x

    def get_next_sample(self):
        x_prime = self.proposal_distribution.sample(self.x)

        log_pdf_x_prime = self.distribution.log_pdf(x_prime)
        log_pdf_x = self.distribution.log_pdf(self.x)

        acceptance_ratio = np.exp(log_pdf_x_prime - log_pdf_x)

        u = np.random.sample()
        if u < acceptance_ratio:
            # accept
            self.x = x_prime
            return self.x
        else:
            # reject
            return self.x


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dim = 2
    a = [np.ones(2)*0, np.ones(2)*5]
    b = [np.eye(2), np.eye(2) * 2]
    distribution = MixtureOfGaussians(a, b, [0.2, 0.8])

    class GaussianSampler(object):
        def __init__(self, dim):
            self.dim = dim
        def sample(self, x):
            return np.random.randn(self.dim) + x
    sampler = MetropolisSampling(dim, distribution, GaussianSampler(dim))

    # burn in
    for _ in range(1000):
        sampler.get_next_sample()

    samples = []
    for i in range(10000):
        sample = sampler.get_next_sample()
        samples.append(sample)

    samples = np.array(samples)

    hisogram = np.histogram2d(samples[:, 0], samples[:, 1], bins=[20, 20], density=True)

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.subplot(2, 1, 2)
    plt.hist2d(samples[:, 0], samples[:, 1], bins=[20, 20])
    plt.show()
