"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import pytest
import numpy as np

pd.reset_option('display.max_rows')
pd.set_option('expand_frame_repr', False)


digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

#SVM Hyperparameters
gamma = [0.001,0.005,0.001,0.0005,0.0001]
c_value=[0.2,0.3,0.7,1,3,5,7,10]

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

#Train test Split
train_fr=0.8
test_fr=0.1
dev_fr=0.1
dev_test_fr=test_fr+dev_fr

# Split data into 80% train and 20% dev/test subsets
X_train, X_dev, y_train, y_dev = train_test_split(data, digits.target, test_size=dev_test_fr, shuffle=True)

# Split data into 50% dev and 50% test subsets
X_dev, X_test, y_dev, y_test = train_test_split(X_dev, y_dev, test_size=dev_fr/dev_test_fr, shuffle=False)

gamm = 0.001
cost = 3

clf = svm.SVC(gamma=gamm,C=cost)
clf.fit(X_train, y_train)

predicted_train = clf.predict(X_train)
predicted_dev = clf.predict(X_dev)
predicted_test = clf.predict(X_test)

acc_train=round(accuracy_score(y_train, predicted_train),2)
acc_val=round(accuracy_score(y_dev, predicted_dev),2)
acc_test=round(accuracy_score(y_test, predicted_test),2)

print('Train Accuracy: ', acc_train)

print('Validation Accuracy: ', acc_val)

print('Testing set Accuracy: ',acc_test)
#Q3
print(len(np.unique(predicted_test)))
print(len(np.unique(y_test)))
try:
    print("Q3")
    assert len(np.unique(predicted_test)) > 1, "a classifier not completely biased to predicting all samples in to one class"
except AssertionError as msg:
    print(msg)
#Q4
try:
    print("Q4")
    assert len(np.unique(predicted_test)) == len(np.unique(y_test)), "a classifier predicts all classes in other words given n number of test samples, the union set of all the predicted labels is same as set of all groundtruth labels"
except AssertionError as msg:
    print(msg)
