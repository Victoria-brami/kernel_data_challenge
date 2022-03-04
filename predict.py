import numpy as np
import pandas as pd
import argparse
from load import Data
from utils import get_classifier, get_feature_extractor

# Xtr = np.array(pd.read_csv('Xtr.csv',header=None,sep=',',usecols=range(3072)))
# Xte = np.array(pd.read_csv('Xte.csv',header=None,sep=',',usecols=range(3072)))
# Ytr = np.array(pd.read_csv('Ytr.csv',sep=',',usecols=[1])).squeeze()


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', default='svm')
    parser.add_argument('--feature_extractor', default='hog')
    parser.add_argument('--output_file', default='Yte_pred.csv')
    return parser.parse_args()


def predict(args):

    # get data
    data = Data()
    Xtr = data.Xtr
    Ytr = data.Ytr
    Xte = data.Xte

    # get the feature extractor
    if args.feature_extractor is not None:
        feature_extractor = get_feature_extractor(args.feature_extractor)
        train_features = feature_extractor.compute(Xtr)
        test_features = feature_extractor.compute(Xtr)
    else:
        train_features = Xtr
        test_features = Xte

    # Get the classifier
    classifier = get_classifier(args.modelname)
    classifier.fit(Xtr, Ytr.reshape(-1))
    Yte = classifier.predict(Xte)

    Yte = {'Prediction': Yte}
    dataframe = pd.DataFrame(Yte)
    dataframe.index += 1
    dataframe.to_csv(args.output_file, index_label='Id')




# define your learning algorithm here
# for instance, define an object called ``classifier''
# classifier.train(Ytr,Xtr)


# predict on the test data
# for instance, Yte = classifier.fit(Xte)

