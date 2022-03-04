import numpy as np
from scipy import optimize
from scipy.linalg import cho_factor, cho_solve
from kernels import *

class KernelSVC:

    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
        self.f_weights = None

    def fit(self, X, y):
        #### You might define here any variable needed for the rest of the code
        N = len(y)

        # Lagrange dual problem
        def loss(alpha):
            # '''--------------dual loss ------------------ '''
            return 0.5 * (alpha * y).T @ self.kernel(X, X) @ (alpha * y) - np.sum(alpha)

            # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            # '''----------------partial derivative of the dual loss wrt alpha-----------------'''
            return np.diag(y) @ self.kernel(X, X) @ np.diag(y) @ alpha - np.ones(N)

            # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0

        # '''----------------function defining the equality constraint------------------'''
        fun_eq = lambda alpha: alpha.T @ y
        # '''----------------jacobian wrt alpha of the  equality constraint------------------'''
        jac_eq = lambda alpha: y

        # '''---------------function defining the ineequality constraint-------------------'''
        fun_ineq1 = lambda alpha: self. C *np.ones(N) - alpha
        # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''
        jac_ineq1 = lambda alpha: - np.eye(N)

        # '''---------------function defining the ineequality constraint-------------------'''
        fun_ineq2 = lambda alpha: alpha
        # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''
        jac_ineq2 = lambda alpha: np.eye(N)

        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq',
                        'fun': fun_ineq1 ,
                        'jac': jac_ineq1},
                       {'type': 'ineq',
                        'fun': fun_ineq2 ,
                        'jac': jac_ineq2},)

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N),
                                   method='SLSQP',
                                   jac=lambda alpha: grad_loss(alpha),
                                   constraints=constraints)
        self.alpha = optRes.x

        ## Assign the required attributes
        supportIndices = self.alpha > self.epsilon
        # '''------------------- A matrix with each row corresponding to a support vector ------------------'''
        self.support = X[supportIndices]
        # '''------------------- weights to compute f ------------------'''
        self.f_weights = (self.alpha *y)[supportIndices]
        # ''' -----------------offset of the classifier------------------ '''
        self.b = np.mean((y - self.kernel(X, X) @ (self.alpha *y))[supportIndices])
        # '''------------------------RKHS norm of the function f ------------------------------'''
        self.norm_f = np.sqrt((self.alpha * y).T @ self.kernel(X, X) @ (self.alpha * y))

    ### Implementation of the separting function $f$
    def separating_function(self ,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return self.kernel(x, self.support) @ self.f_weights


    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * ( d + self. b> 0) - 1