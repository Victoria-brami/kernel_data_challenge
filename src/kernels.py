import numpy as np

class Kernel:
    """ Defines abstract class """
    def __init__(self):
        pass

    def kernel(self, X, Y):
        raise NotImplementedError()

class LinearKernel(Kernel):

    def __init__(self):
        super().__init__()

    def kernel(self, X, Y):
        """ linear kernel : k(x,y) = <x,y> \\
        x, y: array (n_features,)
        """
        return np.dot(X, Y.T)

class PolynomialKernel(Kernel):

    def __init__(self, degree=5):
        super().__init__()
        self.d = degree

    def kernel(self, X, Y):
        """ linear kernel : k(x,y) = <x,y> \\
        x, y: array (n_features,)
        """
        return np.dot(X, Y.T)**self.d


class GaussianKernel(Kernel):

    def __init__(self, sigma=1):
        super().__init__()
        self.sigma = sigma

    def kernel(self, X, Y):
        XX = np.sum(X ** 2, axis=-1)
        YY = np.sum(Y ** 2, axis=-1)
        ## Matrix of shape NxM
        return np.exp(-(XX[:, None] + YY[None, :] - 2 * np.dot(X, Y.T)) / (2 * self.sigma ** 2))











