from models.svm import *
from feature_extractor.hog import HOG
from kernels import *

def get_kernel(args, dev0=None):
    if args.kernel == 'rbf':
        return GaussianKernel(sigma=args.sigma, dev0=dev0)
    elif args.kernel == 'poly':
        return PolynomialKernel(degree=args.degree)
    elif args.kernel == 'linear':
        return LinearKernel()


def get_classifier(args, dev0=None):
    kernel = get_kernel(args, dev0=dev0)
    return MultipleClassSVM(kernel=kernel, C=args.c, type=args.classifier_type, dev0=dev0)

def get_feature_extractor(args):
    return HOG(pixels_per_cell=args.feature_extractor_cell_size, cells_per_block=args.feature_extractor_cells_per_block)


def get_accuracy(preds, labels, verbose=False):
    accuracy = np.mean(preds==labels)
    if verbose:
        print("Accuracy on the set: {:.3f} %".format(accuracy*100))
    return accuracy*100
