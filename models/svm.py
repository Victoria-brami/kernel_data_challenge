import numpy as np
from scipy import optimize
from scipy.linalg import cho_factor, cho_solve
from kernels import *
from load import Data
import copy
from feature_extractor.hog import HOG

class KernelSVC:

    def __init__(self,  kernel, C = 100., epsilon = 1e-3):
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


class MultipleClassSVM:

    def __init__(self,  kernel, C, epsilon=1e-3, type='ova', num_classes=10):
        self.num_classes = num_classes
        self.type = 'non-linear'
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
        self.f_weights = None
        self.classifier_type = type
        self.init_binary_svms()

    def init_binary_svms(self):

        # One class versus all : num_classes SVMs
        if self.classifier_type == 'ova':
            self.binary_classifiers = [KernelSVC(kernel=self.kernel, C=self.C) for i in range(self.num_classes)]

        # One class versus another : num_classes * (num_classes - 1) / 2 SVMs
        else:
            self.binary_classifiers = []
            for i in range(self.num_classes):
                classifiers_i = []
                for j in range(i + 1, self.num_classes):
                    binary_svm_ij = KernelSVC(kernel=self.kernel, C=self.C)
                    classifiers_i.append(binary_svm_ij)
                self.binary_classifiers.append(classifiers_i)



    def fit(self, X, y):

        # One class versus all
        if self.classifier_type == 'ova':
            y_c = copy.deepcopy(y)
            for i in range(self.num_classes):
                print(" Fitting for class {}".format(i))
                # Change the labels to binary labels
                bin_y = [2*(label==i) - 1 for label in y_c]
                # Get line separator for each classifier
                self.binary_classifiers[i].fit(X, bin_y)
        else:
            # One class versus another
            y_c = copy.deepcopy(y)
            for i in range(self.num_classes):
                for j in range(i+1, self.num_classes):
                    print(" Fitting class {} Versus class {}".format(i, j))
                    X_c = X[np.where((y==i) | (y==j))]
                    y_c = y[np.where((y==i) | (y==j))]
                    y_c[np.where(y_c==i)] = 1
                    y_c[np.where(y_c==j)] = -1
                    self.binary_classifiers[i, j-(i+1)].fit(X_c, y_c)

    def separating_function(self, X):
        if self.classifier_type == 'ova':
            n_samples = X.shape[0]
            predictions = np.zeros((self.num_classes, n_samples))
            for i in range(self.num_classes):
                predictions[i, :] = self.kernel(X, self.binary_classifiers[i].support) @ self.binary_classifiers[i].f_weights + self.binary_classifiers[i].b
        else:
            predictions = np.zeros((self.num_classes, self.num_classes))
            for i in range(self.num_classes):
                for j in range(i+1, self.num_classes):
                    predictions[i, j] = self.kernel(X, self.binary_classifiers[i, j-(i+1)].support) @ self.binary_classifiers[i, j-(i+1)].f_weights
                    predictions[i, j-(i+1)] +=  self.binary_classifiers[i, j-(i+1)].b
        return predictions


    def predict(self, X):
        preds = self.separating_function(X)
        if self.classifier_type == 'ova':
            print(preds)
            return np.argmax(preds, axis=0)
        else:
            reshape_preds = copy.deepcopy(preds)
            reshape_preds[preds<0] = -1
            reshape_preds[preds>0] = 1




if __name__ == '__main__':
    data = Data()
    Xtr = data.Xtr[:100]
    Ytr = data.Ytr[:100]
    Xte = data.Xte[:10]

    hog = HOG(pixels_per_cell=4, cells_per_block=3, orientations=9)
    #Xtr = hog._compute_grey_features(data.grey_Xtr_im[:100])

    sigma=1.
    print(Ytr.shape)
    kernel = RBF(sigma).kernel
    # kernel = Linear().kernel
    #kernel = Polynomial(degree=5).kernel
    C = 0.1
    classifier =  MultipleClassSVM(kernel=kernel, C=C, num_classes=4)
    #classifier.fit(Xtr, Ytr.reshape(-1))
    #Yte = classifier.predict(Xtr)
    #print("Accuracy train: ", np.mean(Yte==Ytr))

    import matplotlib.pyplot as plt

    colors = ["blue", "red", "green", "orange"]


    class_1 = np.random.multivariate_normal([0, 0], 2*np.eye(2), 100)
    class_2 = np.random.multivariate_normal([-4, 3], 3/2 * np.eye(2), 100)
    class_3 = np.random.multivariate_normal([3, -2], 2 * np.eye(2), 100)
    class_4 = np.random.multivariate_normal([2, 4],  np.eye(2), 100)
    Y = np.zeros(400)
    Y[100:200] = 1
    Y[200:300] = 2
    Y[300:400] = 3
    X = np.concatenate((class_1, class_2, class_3, class_4))
    rd_perm = np.random.permutation(400)
    X = X[rd_perm]
    Y = Y[rd_perm]

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=[colors[int(i)] for i in Y])
    plt.title("Training set")


    classifier.fit(X, Y)
    Yte = classifier.predict(X)
    print("Accuracy train: ", np.mean(Yte==Y))

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=[colors[int(i)] for i in Yte])
    plt.title("Predictions")
    plt.show()
