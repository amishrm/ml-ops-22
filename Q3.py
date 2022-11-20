# Author: Amit
# Standard scientific Python imports
import matplotlib.pyplot as plt
from tabulate import tabulate
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
import pandas as pd
from sklearn.model_selection import ParameterGrid
import numpy as np
import warnings
from IPython.display import display
import json
from joblib import dump, load
import argparse
warnings.filterwarnings("ignore")

pd.reset_option('display.max_rows')
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)


def test_seed():
    assert seed==51

def preprocess_digits(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    label = dataset.target
    return data, label

def macro_f1(y_true, y_pred, pos_label=1):
    return f1_score(y_true, y_pred, pos_label=pos_label, average='macro', zero_division='warn')

def train_dev_test_split(data, label, train_frac, dev_frac, seed):
    dev_test_frac = 1 - train_frac
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, random_state = seed
    )
    x_test, x_dev, y_test, y_dev = train_test_split(
        x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, random_state = seed
    )

    return x_train, y_train, x_dev, y_dev, x_test, y_test

if __name__ == "__main__":
    train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
    assert train_frac + dev_frac + test_frac == 1.0

    digits = datasets.load_digits()
    data, label = preprocess_digits(digits)
    del digits

    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--clf_name', help='Classifier Name', required=True)
    parser.add_argument('-r','--random_state', help='Random State', required=True)
    args = parser.parse_args()

    args = vars(parser.parse_args())
    clf_name = args['clf_name']
    seed = int(args['random_state'])
    metric_list = [accuracy_score, macro_f1]
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(data, label, train_frac, dev_frac, seed)

    results = {}
    model_path = None
    if clf_name == 'svm' or clf_name == 'SVM':
        best_param = {'gamma':0.001, 'C':0.7}
        clf = svm.SVC(**best_param)
        clf.fit(x_train, y_train)

        predicted = clf.predict(x_test)
        if not clf_name in results:
            results[clf_name]=[]    

        results[clf_name].append({m.__name__:m(y_pred=predicted, y_true=y_test) for m in metric_list})

        best_param_config = ""
        for i in best_param:
            best_param_config += ""+i+"="+str(best_param[i])+"_"
        print("Saving models..")
        best_model_name = clf_name + "_" + best_param_config + ".joblib"
        if model_path == None:
            model_path = 'models/'+best_model_name
        dump(clf, model_path)

        best_model_results = clf_name + "_" + best_param_config + ".txt"
        print("Saving output..")
        with open('results/'+best_model_results, mode='w') as f:
            f.write("test accuracy: {}".format(accuracy_score(y_pred=predicted, y_true=y_test)))
            f.write('\n')
            f.write("test macro-f1: {}".format(macro_f1(y_pred=predicted, y_true=y_test)))
            f.write('\n')
            f.write("model saved at ./{}".format(model_path))


    if clf_name == 'tree':
        best_param =  {'max_features': 'auto', 'max_depth': 10}
        clf = DecisionTreeClassifier(**best_param)
        clf.fit(x_train, y_train)

        predicted = clf.predict(x_test)
        if not clf_name in results:
            results[clf_name]=[]    

        results[clf_name].append({m.__name__:m(y_pred=predicted, y_true=y_test) for m in metric_list})

        best_param_config = ""
        for i in best_param:
            best_param_config += ""+i+"="+str(best_param[i])+"_"

        best_model_name = clf_name + "_" + best_param_config + ".joblib"
        if model_path == None:
            model_path = 'models/'+best_model_name
        dump(clf, model_path)

        best_model_results = clf_name + "_" + best_param_config + ".txt"
        print("Saving output..")
        with open('results/'+best_model_results, mode='w') as f:
            f.write("test accuracy: {}".format(accuracy_score(y_pred=predicted, y_true=y_test)))
            f.write('\n')
            f.write("test macro-f1: {}".format(macro_f1(y_pred=predicted, y_true=y_test)))
            f.write('\n')
            f.write("model saved at ./{}".format(model_path))
    
    print(results)