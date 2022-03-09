from models.svm import *
from feature_extractor.hog import HOG
from kernels import *

def get_kernel(args):
    if args.kernel == 'rbf':
        return RBF(sigma=args.sigma).kernel
    elif args.kernel == 'poly':
        return Polynomial(degree=args.degree).kernel
    elif args.kernel == 'linear':
        return Linear().kernel


def get_classifier(args):
    if args.modelname == 'svm':
        kernel = get_kernel(args)
        return MultipleClassSVM(kernel=kernel, C=args.c)
    kernel = get_kernel(args)
    return MultipleClassSVM(kernel=kernel, C=args.c)

def get_feature_extractor(args):
    if args.feature_extractor == 'hog':
        return HOG(pixels_per_cell=args.feature_extractor_cell_size, cells_per_block=args.feature_extractor_cells_per_block)


def get_accuracy(preds, labels, verbose=False):
    accuracy = np.mean(preds==labels)
    if verbose:
        print("Accuracy on the set: {:.3f} %".format(accuracy*100))
    return accuracy*100
