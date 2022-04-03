import cupy
import cvxopt
import numpy as np
import time
import copy


dev0 = cupy.cuda.Device(0)
dev0.use()

class SVM:
    """
    Implements Binary SVM class
    """
    def __init__(self, kernel, C=1.0, tol_support_vectors=1e-4, dev0=dev0):
        """

        :param kernel: (Kernel) type of the kernel
        :param C: (float)
        :param tol_support_vectors:
        """
        self.kernel = kernel
        self.C = C
        self.tol_support_vectors = tol_support_vectors
        self.dev0 = dev0

    def fit(self, X, y):
        start_time = time.time()
        with self.dev0:
            self.X_train = X
            self.n_train_samples = X.shape[0]
            print("Computing the kernel...")
            self.kernel_X_train = self.kernel.kernel(X, X)
            print("Done!")

            # Quadratic optimization problem
            P = self.kernel_X_train
            q = -y.astype('float')
            y_shape = cupy.squeeze(y).shape[0]
            G = cupy.zeros((2 * y_shape, y_shape))
            G[:y_shape, :] = cupy.diag(cupy.squeeze(y).astype('float'))
            G[y_shape:, :] = -cupy.diag(cupy.squeeze(y).astype('float'))
            h = cupy.concatenate((self.C * cupy.ones(self.n_train_samples), cupy.zeros(self.n_train_samples )))

            # Solve the problem
            # With cvxopt
            P = cvxopt.matrix(P.get())
            q = cvxopt.matrix(q.get())
            G = cvxopt.matrix(G.get())
            h = cvxopt.matrix(h.get())
            #A = cvxopt.matrix(1.0, (1, self.n_train_samples))
            #b = cvxopt.matrix(0.0)
            res = cvxopt.solvers.qp(P=P, q=q, G=G, h=h) #, A=A, b=b)
            x = res["x"]
            self.alphas = cupy.squeeze(cupy.array(x))

            # Retrieve the support vectors
            self.support_vectors_indices = cupy.squeeze(cupy.abs(cupy.array(x))) > self.tol_support_vectors
            self.alphas = self.alphas[self.support_vectors_indices]
            self.support_vectors = self.X_train[self.support_vectors_indices]

            print("Optimization took {:.2f} secs.".format(time.time() - start_time))
            print("Found {} / {} support vectors".format(len(self.support_vectors), self.n_train_samples))

            return self.alphas

    def separating_function(self, X):
        """
        X: array (n_samples, n_features)\\
        Return: float array (n_samples,)
        """
        with self.dev0:
            K = self.kernel.kernel(X, self.support_vectors)
            y = np.dot(K, self.alphas)
            return y

    def predict(self, X):
        y = self.separating_function(X)
        return cupy.where(y > 0, 1, -1)


class MultipleClassSVM:

    def __init__(self, kernel, C, epsilon=1e-3, type='ova', num_classes=10, cross_val_folds=5, dev0=None):
        self.num_classes = num_classes
        self.type = 'non-linear'
        self.C = C
        self.kernel = kernel
        self.epsilon = epsilon
        self.classifier_type = type
        self.cross_val_folds = cross_val_folds
        self.init_binary_svms()
        self.dev0 = dev0

    def init_binary_svms(self):
        # One class versus all : num_classes SVMs
        if self.classifier_type == 'ova':
            self.binary_classifiers = []
            for i in range(self.num_classes):
                    self.binary_classifiers.append(SVM(kernel=self.kernel, C=self.C, dev0=self.dev0))

        # One class versus another : num_classes * (num_classes - 1) / 2 SVMs
        elif self.classifier_type == 'ovo':
            self.binary_classifiers = []
            for i in range(self.num_classes):
                classifiers_i = []
                for j in range(i + 1, self.num_classes):
                    classifiers_i.append(SVM(kernel=self.kernel, C=self.C))
                self.binary_classifiers.append(classifiers_i)


    def fit(self, X, y):

        # One class versus all
        if self.classifier_type == 'ova':
            y_c = copy.deepcopy(y)
            for i in range(self.num_classes):
                print(" Fitting for class {}".format(i))
                # Change the labels to binary labels
                bin_y = cupy.array([2 * (label == i) - 1 for label in y_c])
                # Get line separator for each classifier
                self.binary_classifiers[i].fit(X, bin_y)
        else:
            # One class versus another
            for i in range(self.num_classes):
                for j in range(i + 1, self.num_classes):
                    print(" Fitting class {} Versus class {}".format(i, j))
                    used_ids = np.where((y == i) | (y == j))[0]
                    X_c = copy.deepcopy(X)[used_ids]
                    y_c = copy.deepcopy(y)[used_ids]
                    y_c[y_c == i] = 1
                    y_c[y_c == j] = -1
                    self.binary_classifiers[i][j - (i + 1)].fit(X_c, y_c)

    def separating_function(self, X):
        with self.dev0:
            if self.classifier_type == 'ova':
                n_samples = X.shape[0]
                predictions = cupy.zeros((self.num_classes, n_samples))
                for i in range(self.num_classes):
                    predictions[i, :] = self.binary_classifiers[i].separating_function(X)
            else:
                n_samples = X.shape[0]
                predictions = cupy.zeros((n_samples, self.num_classes, self.num_classes))
                for i in range(self.num_classes):
                    for j in range(i + 1, self.num_classes):
                        predictions[:, i, j] = self.binary_classifiers[i][j - (i + 1)].predict(X)
                        predictions[:, j, i] = - predictions[:, i, j]
            return predictions

    def predict(self, X):
        with self.dev0:
            preds = self.separating_function(X)
            if self.classifier_type == 'ova':
                probas = cupy.exp(-(preds - cupy.min(preds, axis=0))) / cupy.sum(
                    cupy.exp(-(preds - cupy.min(preds, axis=0))), axis=0)
                print(probas)
                return cupy.argmax(preds, axis=0), probas
            else:
                preds = self.separating_function(X)
                preds_over_classes = cupy.sum(preds, axis=2)
                probas = None
                return cupy.argmax(preds_over_classes, axis=1), probas