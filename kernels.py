import numpy as np


class Linear:
    """ Implements the lineal kernel """
    def __init__(self):
        self.one = 1

    def kernel(self, X, Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        return X @ Y.T

class RBF:
    """ Implements the Gaussian kernel """
    def __init__(self, sigma=1.):
        self.sigma = sigma  ## the variance of the kernel

    def kernel(self, X, Y):
        XX = np.sum(X ** 2, axis=-1)
        YY = np.sum(Y ** 2, axis=-1)
        ## Matrix of shape NxM
        return np.exp(-(XX[:, None] + YY[None, :] - 2 * np.dot(X, Y.T)) / (2 * self.sigma ** 2))

