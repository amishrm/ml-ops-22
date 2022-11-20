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
from numpy import random
warnings.filterwarnings("ignore")

pd.reset_option('display.max_rows')
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

def preprocess_digits(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    label = dataset.target
    return data, label

def test_seed():
    assert seed==51

def train_dev_test_split(data, label, train_frac, dev_frac, seed):
    dev_test_frac = 1 - train_frac
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, random_state = seed, shuffle=True
    )
    x_test, x_dev, y_test, y_dev = train_test_split(
        x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, random_state = seed, shuffle=True
    )

    return x_train, y_train, x_dev, y_dev, x_test, y_test

if __name__ == "__main__":
    train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
    assert train_frac + dev_frac + test_frac == 1.0

    digits = datasets.load_digits()
    data, label = preprocess_digits(digits)
    del digits

    print("Q1.a. We can pass same value in random_state variable of train_test_split function. \n we need to set random.seed and generate a new random varibale and pass this generated variable in train_test_split function.")

    print("Q1.b.")
    random.seed(42)
    seed = random.randint(100)
    test_seed()
    print("Test case is successful.")
    print("")
    print("Q1.c.")
    seed = random.randint(10)
    test_seed()
    print("Test case is failed.")


    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(data, label, train_frac, dev_frac, seed)
