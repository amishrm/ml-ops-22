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

#Define Best accuracy 
best_acc = 0.
val_test_acc = 0.
best_params = {}
results = []
best_result = []

# Create a classifier: a support vector classifier
print("Model Training in progress..................\n")
for i in gamma:
    for j in c_value:
        clf = svm.SVC(gamma=i,C=j)
        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        predicted_train = clf.predict(X_train)
        predicted_dev = clf.predict(X_dev)
        predicted_test = clf.predict(X_test)

        acc_train=round(accuracy_score(y_train, predicted_train),2)
        acc_val=round(accuracy_score(y_dev, predicted_dev),2)
        acc_test=round(accuracy_score(y_test, predicted_test),2)

        parameter = "Gamma: {}, C: {}".format(i,j)
        results.append({'Hyper parameter':parameter,'Train Accuracy':acc_train,'Dev Accuracy': acc_val, 'Test Accuracy' : acc_test})

        print("[ Gamma: {}, C: {} ] ===> Dev Accuracy : {} , Test Accuracy : {} ".format(i,j,acc_val,acc_test))
        if acc_test>best_acc:
            best_acc=acc_test
            best_params={"Gamma":i ,"C":j}
            val_test_acc={"val_accuract: ":acc_val, "Test_accuracy: ":acc_test}
            best_result = [{'Hyper parameter':"Gamma: {}, C: {}".format(i,j),'Train Accuracy':acc_train,'Dev Accuracy': acc_val, 'Test Accuracy' : best_acc}]

print("\n")
print("------------------Best Results ----------------------------------")
print(val_test_acc)
print("\n----------------------------------------------------------------")
print("Best Parms Gamma: {} , C: {}".format(best_params["Gamma"],best_params["C"]))

results_df = pd.DataFrame.from_dict(results)
best_result_df = pd.DataFrame.from_dict(best_result)

print("\n------------------------- Report ---------------------------")
print(results_df)
print("\n------------------------- Best Model Report --------------------------")
print(best_result_df)