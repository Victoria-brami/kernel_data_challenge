import numpy as np
import cupy
import pandas as pd
import argparse
from load import Data
from utils import get_classifier, get_feature_extractor, get_accuracy

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
    parser.add_argument('--feature_extractor', default='hog', choices=[None, 'hog'])
    parser.add_argument('--feature_extractor_cell_size', default=8)
    parser.add_argument('--feature_extractor_cells_per_block', default=3)
    parser.add_argument('--output_file', default='results/Yte_pred.csv')
    parser.add_argument('--train_file', default='results/Ytrain_pred.csv')
    return parser.parse_args()


def predict(args, dev0=None):

    data = Data(repository=args.datapath)
    Xtr = data.Xtr
    Ytr = data.Ytr
    Xte = data.Xte

    Xtr_im = data.Xtr_im
    Xte_im = data.Xte_im

    # get the feature extractor
    if args.feature_extractor is not None:
        train_feature_extractor = get_feature_extractor(args)
        train_features = train_feature_extractor._compute_features(Xtr_im)
        test_feature_extractor = get_feature_extractor(args)
        test_features = test_feature_extractor._compute_features(Xte_im)
    else:
        train_features = Xtr
        test_features = Xte
        
    train_features = cupy.array(train_features)
    test_features = cupy.array(test_features)
    Ytr = cupy.array(Ytr)

    # Get the classifier
    classifier = get_classifier(args, dev0)

    # Train the classifier
    classifier.fit(train_features, Ytr.reshape(-1))
    Y_train_preds, pred_probas = classifier.predict(train_features)
    get_accuracy(Y_train_preds, Ytr, verbose=True)
    Ytrain = {'Prediction': Y_train_preds.get(), 'Labels': Ytr.get()}
    dataframe = pd.DataFrame(Ytrain)
    dataframe.index += 1
    dataframe.to_csv(args.train_file, index_label='Id')

    # make the predictions
    Yte, _ = classifier.predict(test_features)
    Yte = {'Prediction': Yte.get()}
    dataframe = pd.DataFrame(Yte)
    dataframe.index += 1
    dataframe.to_csv(args.output_file, index_label='Id')




if __name__ == '__main__':
    args = parser()
    print(args)
    predict(args)

