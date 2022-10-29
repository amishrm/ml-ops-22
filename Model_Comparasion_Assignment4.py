# Author: Amit
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import ParameterGrid
import numpy as np
import warnings
warnings.filterwarnings("ignore")

pd.reset_option('display.max_rows')
pd.set_option('expand_frame_repr', False)


digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

#train/test/validation dataset splits.
train_frac_split=[0.8,0.7,0.6,0.4,0.5]
dev_frac_split=[0.1,0.1,0.1,0.2,0.2]

def train_dev_test_split(X, y, split):
    #Train test Split
    dev_test_fr = 1-train_frac_split[split]
    dev_fr = dev_frac_split[split]
    
    # Split data into 80% train and 20% dev/test subsets
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=dev_test_fr, shuffle=True)

    # Split data into 50% dev and 50% test subsets
    X_dev, X_test, y_dev, y_test = train_test_split(X_dev, y_dev, test_size=dev_fr/dev_test_fr, shuffle=False)

    return X_train, y_train, X_dev, y_dev, X_test, y_test

def train(X, y, classifier, params, n_split = 5):
    best_result = pd.DataFrame()
    param_grid = ParameterGrid(params)

    for i in range(n_split):
        X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(X, y, split=i)

        #Define Best accuracy 
        best_acc = 0.
        results = []
        
        # Create a classifier: a DT classifier
        for param in param_grid:
            clf = classifier(**param)
            # Learn the digits on the train subset
            clf.fit(X_train, y_train)
            # Predict the value of the digit on the test subset
            predicted_train = clf.predict(X_train)
            predicted_dev = clf.predict(X_dev)
            predicted_test = clf.predict(X_test)

            acc_train=round(accuracy_score(y_train, predicted_train),2)
            acc_val=round(accuracy_score(y_dev, predicted_dev),2)
            acc_test=round(accuracy_score(y_test, predicted_test),2)

            print("Run: {}, Split Ratio: ({},{}), Hyper parameter: {}, Train Accuracy: {}, Test Accuracy: {}".format(i+1, train_frac_split[i],dev_frac_split[i], param, acc_train, acc_test))
            if acc_test > best_acc:
                best_acc=acc_test
                results.append({'Run':i+1,'Split Ratio':(train_frac_split[i],dev_frac_split[i]),'Hyper parameter':param,'Train Accuracy':acc_train,'Dev Accuracy': acc_val, 'Test Accuracy' : best_acc})
                
        results_df = pd.DataFrame.from_dict(results)
        final_df = results_df.sort_values(by=['Test Accuracy'], ascending=False).iloc[:1]
        best_result = best_result.append(final_df, ignore_index=True)

    print("********************Printing Best Result*********************")   
    print(best_result)
    return best_result

print("Model Training in progress..................\n")
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
n_fold = 5

#SVM Hyperparameters
print("Train SVM model and below are results..")
svm_params = {'gamma' : [0.001,0.005,0.001,0.0005,0.0001],
            'C':[0.2,0.3,0.7,1,3,5,7,10]
         }
SVMResult = train(data, digits.target, svm.SVC, svm_params, n_fold)

print("")
print("")

#DT Hyperparameters
print("Train Decision Tree model and below are results..")
dt_params = {'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth' : [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
              'criterion' :['gini', 'entropy']
             }
DTResult = train(data, digits.target, DecisionTreeClassifier, dt_params, n_fold)
print("")
print("")
print("**********SVM Stats************")
print(SVMResult.describe())
print("")
print("**********Decision Tree Stats************")
print(DTResult.describe())

print("")
print("")
print('SVM, Mean Accuracy: {}, Standard Deviation of Accuracy: {}'.format(SVMResult['Test Accuracy'].mean(), SVMResult['Test Accuracy'].std()))

print("")
print("")
print('Decision Tree, Mean Accuracy: {}, Standard Deviation of Accuracy: {}'.format(DTResult['Test Accuracy'].mean(), DTResult['Test Accuracy'].std()))


print("")
print("")


print("**********Performance Metric for SVM and Decision Tree***********")
finalResultDf = pd.merge(SVMResult[['Run','Test Accuracy']],DTResult[['Run','Test Accuracy']],how='left', left_on=['Run'], right_on = ['Run']).rename(columns={'Test Accuracy_x': 'svm', 'Test Accuracy_y': 'decision_tree'})


stats = [{'Run':'mean','svm':finalResultDf['svm'].mean(),'decision_tree':finalResultDf['decision_tree'].mean()},
         {'Run':'std','svm':finalResultDf['svm'].std(),'decision_tree':finalResultDf['decision_tree'].std()}]
tempDf = pd.DataFrame.from_dict(stats)

finalResult = pd.concat([finalResultDf, tempDf], ignore_index=True)

print(finalResult.to_string(index=False))