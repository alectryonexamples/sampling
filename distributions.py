import numpy as np
import copy
import scipy.special

class ContinuousDistributionBase(object):
    """
    @brief      Base Class for continuous distributions.
    """  
    def __init__(self):
        pass

    def dim(self):
        """
        @brief      returns the dimension of the the probability space
        """
        raise Exception("Not yet implemented.")

    def pdf(self, x):
        """
        @brief      computes pdf of distribution
        
        @param      x     np array of length dim()
        """
        raise Exception("Not yet implemented.")

    def log_pdf(self, x):
        """
        @brief      computes log pdf of distribution
        
        @param      x     np array of length dim()
        """
        raise Exception("Not yet implemented.")

class Gaussian(ContinuousDistributionBase):
    def __init__(self, center, cov):
        self.dim = len(center)

        if cov.shape[0] != self.dim or cov.shape[1] != self.dim:
            raise Exception("Covariance must be N by N.")
        self.center = np.copy(center)
        self.cov = np.copy(cov)

        self.cov_inv = np.linalg.inv(cov)
        self.det_inv = np.linalg.det(self.cov_inv)

    def dim(self):
        return self.dim

    def pdf(self, x):
        if len(x) != self.dim:
            raise Exception("x must be same dimension as distribution.")
        
        x_centered = x - self.center
        pdf = np.power(2*np.pi, -self.dim*0.5) * self.det_inv * np.exp(-0.5 * np.dot(np.dot(x_centered, self.cov_inv), x_centered))
        return pdf

    def log_pdf(self, x):
        if len(x) != self.dim:
            raise Exception("x must be same dimension as distribution.")

        x_centered = x - self.center
        log_pdf = np.log(np.power(2*np.pi, -self.dim*0.5)*self.det_inv ) + (-0.5 * np.dot(np.dot(x_centered, self.cov_inv), x_centered))
        return log_pdf

class MixtureOfGaussians(ContinuousDistributionBase):
    def __init__(self, centers, covariances, weights):
        if len(centers) != len(covariances) or len(centers) != len(weights):
            raise Exception("centers must be same length as covariances.")

        self.dim = len(centers[0])

        for center in centers:
            if len(center) != self.dim:
                raise Exception("All centers must be same dim.")

        for cov in covariances:
            if cov.shape[0] != self.dim or cov.shape[1] != self.dim:
                raise Exception("All covariances must be N by N.")

        if abs(np.sum(weights) - 1) > 1e-3:
            raise Exception("Weights must sum to 1.")

        self.gaussians = [Gaussian(center, cov) for center, cov in zip(centers, covariances)]
        self.weights = copy.deepcopy(weights)

    def dim(self):
        return self.dim
        
    def pdf(self, x):
        pdf = np.exp(self.log_pdf(x))
        return pdf

    def log_pdf(self, x):
        if len(x) != self.dim:
            raise Exception("x must be same dimension as distribution.")

        log_list = []
        for gaussian, weight in zip(self.gaussians, self.weights):
            log_list.append(gaussian.log_pdf(x) + np.log(weight))
        log_pdf = scipy.special.logsumexp(log_list)
        return log_pdf

