import cvxopt
import numpy as np
import time
import copy


class SVM:
    """
    Implements Binary SVM class
    """
    def __init__(self, kernel, C=1.0, tol_support_vectors=1e-4):
        self.kernel = kernel
        self.C = C
        self.tol_support_vectors = tol_support_vectors

    def fit(self, X, y):
        start_time = time.time()
        self.X_train = X
        self.n_train_samples = X.shape[0]
        self.kernel_X_train = self.kernel.kernel(X, X)

        # Quadratic optimization problem
        P = cvxopt.matrix(np.einsum('ij,i,j->ij', self.kernel_X_train, y, y))
        q = cvxopt.matrix(- np.ones(self.n_train_samples))
        A = cvxopt.matrix(y.astype('float').reshape(1, -1))
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.vstack((-np.eye(self.n_train_samples), 
                                     np.eye(self.n_train_samples))))
        h = cvxopt.matrix(np.hstack((np.zeros(self.n_train_samples), 
                          self.C * np.ones(self.n_train_samples))))
        res = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
        self.alphas = np.ravel(res["x"])
        assert self.alphas.shape == (self.n_train_samples, )

        # Retrieve the support vectors
        margin_idx = (self.alphas > self.tol_support_vectors)
        self.margin_vectors = self.X_train[margin_idx]
        assert self.margin_vectors.shape[1] == self.X_train.shape[1]
        self.w = y[margin_idx] * self.alphas[margin_idx]
        assert self.w.shape == (len(self.margin_vectors), )
        
        boundary_idx = (self.alphas > self.tol_support_vectors) * (self.C - self.alphas > self.tol_support_vectors)
        self.support_vectors = self.X_train[boundary_idx]
        support_preds = self.w.dot(self.kernel.kernel(self.margin_vectors, 
                                                      self.support_vectors))
        self.b = np.mean(y[boundary_idx] - support_preds)
        print(self.b)
        print("Optimization took {:.2f} secs.".format(time.time() - start_time))
        print("Found {} / {} support vectors".format(len(self.support_vectors), 
                                                     self.n_train_samples))

    def separating_function(self, X):
        K = self.kernel.kernel(self.margin_vectors, X)
        return self.w@K + self.b

    def predict(self, X):
        y = self.separating_function(X)
        return np.where(y > 0, 1, -1)

class MultipleClassSVM:

    def __init__(self, kernel, C, epsilon=1e-3, typ_='ova', num_classes=10):
        self.num_classes = num_classes
        self.C = C
        self.kernel = kernel
        self.epsilon = epsilon
        self.classifier_type = typ_
        self.init_binary_svms()

    def init_binary_svms(self):
        # One class versus all : num_classes SVMs
        if self.classifier_type == 'ova':
            self.binary_classifiers = []
            for i in range(self.num_classes):
                    self.binary_classifiers.append(SVM(kernel=self.kernel, C=self.C))

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
                bin_y = np.array([2 * (label == i) - 1 for label in y_c])
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
                    idx_i = (y_c == i)
                    idx_j = (y_c == j)
                    y_c[idx_i] = 1
                    y_c[idx_j] = -1
                    assert np.sum(y_c == 1) >= 1
                    assert np.sum(y_c == -1) >= 1
                    self.binary_classifiers[i][j - (i + 1)].fit(X_c, y_c)

    def separating_function(self, X):
        if self.classifier_type == 'ova':
            n_samples = X.shape[0]
            predictions = np.zeros((self.num_classes, n_samples))
            for i in range(self.num_classes):
                predictions[i, :] = self.binary_classifiers[i].separating_function(X)
        else:
            n_samples = X.shape[0]
            predictions = np.zeros((n_samples, self.num_classes, self.num_classes))
            for i in range(self.num_classes):
                for j in range(i + 1, self.num_classes):
                    predictions[:, i, j] = self.binary_classifiers[i][j - (i + 1)].separating_function(X)
                    predictions[:, j, i] = - predictions[:, i, j]
        return predictions

    def predict(self, X):
        preds = self.separating_function(X)
        if self.classifier_type == 'ova':
            probas = np.exp(-(preds - np.min(preds, axis=0))) / np.sum(
                np.exp(-(preds - np.min(preds, axis=0))), axis=0)
            return np.argmax(preds, axis=0), probas
        else:
            preds_over_classes = np.sum(preds, axis=2)
            probas = None
            return np.argmax(preds_over_classes, axis=1), probas