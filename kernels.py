import numpy as np
import cupy

dev0 = cupy.cuda.Device(0)
dev0.use()

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
        return cupy.dot(X, Y.T)

class PolynomialKernel(Kernel):

    def __init__(self, degree=5):
        super().__init__()
        self.d = degree

    def kernel(self, X, Y):
        """ linear kernel : k(x,y) = <x,y> \\
        x, y: array (n_features,)
        """
        return cupy.dot(X, Y.T)**self.d


class GaussianKernel(Kernel):

    def __init__(self, sigma=1, dev0=None):
        super().__init__()
        self.sigma = sigma
        self.dev0 = dev0

    def kernel(self, X, Y):
        with self.dev0:
          XX = cupy.sum(X ** 2, axis=-1)
          YY = cupy.sum(Y ** 2, axis=-1)
          ## Matrix of shape NxM
          return cupy.exp(-(XX[:, None] + YY[None, :] - 2 * cupy.dot(X, Y.T)) / (2 * self.sigma ** 2))











