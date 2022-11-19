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
warnings.filterwarnings("ignore")

pd.reset_option('display.max_rows')
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

def get_all_combs(param_vals, param_name, combs_so_far):
    new_combs_so_far = []        
    for c in combs_so_far:        
        for v in param_vals:
            cc = c.copy()
            cc[param_name] = v
            new_combs_so_far.append(cc)
    return new_combs_so_far

def get_all_h_param_comb(params):
    h_param_comb = [{}]
    for p_name in params:
        h_param_comb = get_all_combs(
            param_vals=params[p_name], param_name=p_name, combs_so_far=h_param_comb
        )

    return h_param_comb

def preprocess_digits(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    label = dataset.target
    return data, label

def macro_f1(y_true, y_pred, pos_label=1):
    return f1_score(y_true, y_pred, pos_label=pos_label, average='macro', zero_division='warn')

def train_dev_test_split(data, label, train_frac, dev_frac):
    dev_test_frac = 1 - train_frac
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, shuffle=True
    )
    x_test, x_dev, y_test, y_dev = train_test_split(
        x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, shuffle=True
    )

    return x_train, y_train, x_dev, y_dev, x_test, y_test

def h_param_tuning(h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric, verbose=False):
    best_metric = -1.0
    best_model = None
    best_h_params = None
    # 2. For every combination-of-hyper-parameter values
    for cur_h_params in h_param_comb:

        # PART: setting up hyperparameter
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)

        # PART: Train model
        # 2.a train the model
        # Learn the digits on the train subset
        clf.fit(x_train, y_train)

        # print(cur_h_params)
        # PART: get dev set predictions
        predicted_dev = clf.predict(x_dev)

        # 2.b compute the accuracy on the validation set
        cur_metric = metric(y_pred=predicted_dev, y_true=y_dev)

        # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest.
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_model = clf
            best_h_params = cur_h_params
            if verbose:
                print("Found new best metric with :" + str(cur_h_params))
                print("New best val metric:" + str(cur_metric))
    return best_model, best_metric, best_h_params

def tune_and_save(
    clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path):
    best_model, best_metric, best_h_params = h_param_tuning(
        h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric
    )

    # save the best_model
    best_param_config = "_".join(
        [h + "=" + str(best_h_params[h]) for h in best_h_params]
    )

    if type(clf) == svm.SVC:
        model_type = "svm"

    if type(clf) == DecisionTreeClassifier:
        model_type = "decision_tree"

    best_model_name = model_type + "_" + best_param_config + ".joblib"
    if model_path == None:
        model_path = best_model_name
    dump(best_model, model_path)

    print("Best hyperparameters were:" + str(best_h_params))

    print("Best Metric on Dev was:{}".format(best_metric))

    return model_path

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

svm_params = {'gamma' : [0.001,0.005,0.001,0.0005,0.0001],
            'C':[0.2,0.3,0.7,1,3,5,7,10]
         }

dt_params = {'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth' : [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
             }

svm_h_param_comb = get_all_h_param_comb(svm_params)
dec_h_param_comb = get_all_h_param_comb(dt_params)

h_param_comb = {"svm": svm_h_param_comb, "decision_tree": dec_h_param_comb}

print(h_param_comb)

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

data, label = preprocess_digits(digits)
# housekeeping
del digits

metric_list = [accuracy_score, macro_f1]
h_metric = accuracy_score


n_cv = 5
results = {}
for n in range(n_cv):
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac
    )
    # PART: Define the model
    # Create a classifier: a support vector classifier
    models_of_choice = {
        "svm": svm.SVC(),
        "decision_tree": DecisionTreeClassifier(),
    }
    for clf_name in models_of_choice:
        clf = models_of_choice[clf_name]
        print("[{}] Running hyper param tuning for {}".format(n,clf_name))
        actual_model_path = tune_and_save(
            clf, x_train, y_train, x_dev, y_dev, h_metric, h_param_comb[clf_name], model_path=None
        )

        # 2. load the best_model
        best_model = load(actual_model_path)

        # PART: Get test set predictions
        # Predict the value of the digit on the test subset
        predicted = best_model.predict(x_test)
        if not clf_name in results:
            results[clf_name]=[]    

        results[clf_name].append({m.__name__:m(y_pred=predicted, y_true=y_test) for m in metric_list})
        # 4. report the test set accurancy with that best model.
        # PART: Compute evaluation metrics
        print(
            f"Classification report for classifier {clf}:\n"
            f"{metrics.classification_report(y_test, predicted)}\n"
        )

print(results)