import sys
sys.path.append("src/")
import pandas as pd
import argparse
from load import Data
from utils import get_classifier, get_feature_extractor, get_accuracy
from copy import deepcopy
import time

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', default='svm')
    parser.add_argument('--datapath', type=str, default='data/')
    parser.add_argument('--kernel', default='rbf')
    parser.add_argument('--sigma', type=float, default=1)
    parser.add_argument('--degree', type=int, default=5)
    parser.add_argument('--c', type=float, default=1)
    parser.add_argument('--classifier_type', type=str, default='ovo', choices=['ova', 'ovo'])
    parser.add_argument('--feature_extractor', default='hog', choices=['None', 'hog'])
    parser.add_argument('--feature_extractor_cell_size', default=8)
    parser.add_argument('--feature_extractor_cells_per_block', default=3)
    parser.add_argument('--output_file', default='results/Yte_pred.csv')
    parser.add_argument('--train_file', default='results/Ytrain_pred.csv')
    parser.add_argument('--val_split', type=int, default=0)
    return parser.parse_args()


def predict(args):
    
    start_time = time.time()
    data = Data(repository=args.datapath)
    Xtr = data.Xtr
    Ytr = data.Ytr
    Xte = data.Xte

    Xtr_im = data.Xtr_im
    Xte_im = data.Xte_im

    # get the feature extractor
    if args.feature_extractor != "None":
        train_feature_extractor = get_feature_extractor(args)
        train_features = train_feature_extractor.compute_features(Xtr_im)
        test_feature_extractor = get_feature_extractor(args)
        test_features = test_feature_extractor.compute_features(Xte_im)
    else:
        train_features = Xtr
        test_features = Xte
        
    # Validation split
    if args.val_split == 1:
        val_features = deepcopy(train_features[3000:])
        train_features = deepcopy(train_features[1000:])
        Yval = deepcopy(Ytr[:1000])
        Ytr = deepcopy(Ytr[1000:])

    # Get the classifier
    classifier = get_classifier(args)

    # Train the classifier
    classifier.fit(train_features, Ytr.reshape(-1))
    Y_train_preds = classifier.predict(train_features)
    assert get_accuracy(Ytr.ravel(), Ytr.ravel(), verbose=False) == 100
    get_accuracy(Y_train_preds, Ytr.ravel(), verbose=True)
    if args.train_file != "None":
        Ytrain = {'Prediction': Y_train_preds}
        dataframe = pd.DataFrame(Ytrain)
        dataframe.index += 1
        dataframe.to_csv(args.train_file, index_label='Id')
    
    # validation
    if args.val_split == 1:
        Y_val_preds = classifier.predict(val_features)
        get_accuracy(Y_val_preds, Yval.ravel(), verbose=True)

    # make the predictions
    Yte = classifier.predict(test_features)
    Yte = {'Prediction': Yte}
    dataframe = pd.DataFrame(Yte)
    dataframe.index += 1
    dataframe.to_csv(args.output_file, index_label='Id')
    
    print("Total running time : {:.2f} secs.".format(time.time() - start_time))

if __name__ == '__main__':
    args = parser()
    print(args)
    predict(args)

