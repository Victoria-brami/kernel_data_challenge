import numpy as np
import cupy
import pandas as pd
import argparse
from load import Data
from utils import get_classifier, get_feature_extractor, get_accuracy
import sys

cupy.cuda.set_allocator(None)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', default='svm')
    parser.add_argument('--datapath', type=str, default='data/')
    parser.add_argument('--kernel', default='rbf')
    parser.add_argument('--sigma', type=float, default=1)
    parser.add_argument('--degree', type=int, default=5)
    parser.add_argument('--c', type=float, default=1)
    parser.add_argument('--classifier_type', type=str, default='ova', choices=['ova', 'ovo'])
    parser.add_argument('--feature_extractor', default='hog_color', choices=[None, 'hog_color', 'hog_grey'])
    parser.add_argument('--feature_extractor_cell_size', default=8)
    parser.add_argument('--feature_extractor_cells_per_block', default=3)
    parser.add_argument('--output_file', default='Yte_pred.csv')
    return parser.parse_args()


def predict(args):

    print(sys.path[0])
    # get data
    data = Data(repository=args.datapath)
    Xtr = data.Xtr
    Ytr = data.Ytr
    Xte = data.Xte

    Xtr_im = data.Xtr_im
    Xte_im = data.Xte_im

    grey_Xtr_im = data.grey_Xtr_im
    grey_Xte_im = data.grey_Xte_im

    # get the feature extractor
    if args.feature_extractor is not None:
        if args.feature_extractor=='hog_grey':
            train_feature_extractor = get_feature_extractor(args)
            train_features = train_feature_extractor._compute_grey_features(grey_Xtr_im)
            test_feature_extractor = get_feature_extractor(args)
            test_features = test_feature_extractor._compute_grey_features(grey_Xte_im)

        elif args.feature_extractor=='hog_color':
            train_feature_extractor = get_feature_extractor(args)
            print("test", type(train_feature_extractor))
            train_features = train_feature_extractor._compute_features(Xtr_im)
            test_feature_extractor = get_feature_extractor(args)
            test_features = test_feature_extractor._compute_features(Xte_im)
    else:
        train_features = Xtr
        test_features = Xte

    # Get the classifier
    classifier = get_classifier(args)

    # Train the classifier
    classifier.fit(train_features, Ytr.reshape(-1))
    Y_train_preds, pred_probas = classifier.predict(train_features)
    accuracy = get_accuracy(Y_train_preds, Ytr, verbose=True)

    # make the predictions
    Yte = classifier.predict(test_features)
    Yte = {'Prediction': Yte}
    dataframe = pd.DataFrame(Yte)
    dataframe.index += 1
    dataframe.to_csv(args.output_file, index_label='Id')




if __name__ == '__main__':
    args = parser()
    print(args)
    predict(args)

