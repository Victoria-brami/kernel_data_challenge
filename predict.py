import numpy as np
import pandas as pd
import argparse
from load import Data
from utils import get_classifier, get_feature_extractor, get_accuracy

# Xtr = np.array(pd.read_csv('Xtr.csv',header=None,sep=',',usecols=range(3072)))
# Xte = np.array(pd.read_csv('Xte.csv',header=None,sep=',',usecols=range(3072)))
# Ytr = np.array(pd.read_csv('Ytr.csv',sep=',',usecols=[1])).squeeze()


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', default='svm')
    parser.add_argument('--datapath', type=str, default='data/')
    parser.add_argument('--kernel', default='rbf')
    parser.add_argument('--sigma', type=float, default=1)
    parser.add_argument('--degree', type=int, default=5)
    parser.add_argument('--c', type=float, default=1)
    parser.add_argument('--feature_extractor', default='hog')
    parser.add_argument('--feature_extractor_cell_size', default=8)
    parser.add_argument('--feature_extractor_cells_per_block', default=3)
    parser.add_argument('--output_file', default='Yte_pred.csv')
    return parser.parse_args()


def predict(args):

    # get data
    data = Data(repository=args.datapath)
    Xtr = data.Xtr
    Ytr = data.Ytr
    Xte = data.Xte

    Xtr_im = data.Xtr_im
    Xte_im = data.Xte_im

    # get the feature extractor
    if args.feature_extractor is not None:
        feature_extractor = get_feature_extractor(args)
        train_features = feature_extractor._compute_features(Xtr_im)
        test_features = feature_extractor._compute_features(Xte_im)
    else:
        train_features = Xtr
        test_features = Xte

    # Get the classifier
    classifier = get_classifier(args)

    # Train the classifier
    classifier.fit(train_features, Ytr.reshape(-1))
    Y_train_preds = classifier.predict(train_features)
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

