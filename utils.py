from models.svm import KernelSVC
from feature_extractor.hog import HOG

def get_classifier(args):

    if args.modelname == 'svm':
        return KernelSVC(kernel=args.kernel)
    else:
        return KernelSVC(kernel=args.kernel)

def get_feature_extractor(args):
    if args.feature_extractor == 'hog':
        return HOG()