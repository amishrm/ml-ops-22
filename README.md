# Assignment 4

Accuracy for SVM and Decision Tree Classifier for digit classification:

### ********************Printing Best Result for SVM*********************
   Run Split Ratio             Hyper parameter  Train Accuracy  Dev Accuracy  Test Accuracy
    1  (0.8, 0.1)  {'C': 0.3, 'gamma': 0.001}            0.99          0.99           0.99
    2  (0.7, 0.1)  {'C': 0.2, 'gamma': 0.001}            0.99          0.98           0.99
    3  (0.6, 0.1)  {'C': 0.3, 'gamma': 0.001}            0.99          0.97           0.99
    4  (0.4, 0.2)  {'C': 0.7, 'gamma': 0.001}            1.00          0.98           0.99
    5  (0.5, 0.2)    {'C': 3, 'gamma': 0.001}            1.00          0.99           1.00

### ********************Printing Best Result for Decision Tree*********************

   Run Split Ratio                                    Hyper parameter  Train Accuracy  Dev Accuracy  Test Accuracy
    1  (0.8, 0.1)  {'criterion': 'gini', 'max_depth': 35, 'max_fe...            1.00          0.84           0.90
    2  (0.7, 0.1)  {'criterion': 'entropy', 'max_depth': 10, 'max...            0.99          0.81           0.85
    3  (0.6, 0.1)  {'criterion': 'entropy', 'max_depth': 10, 'max...            1.00          0.82           0.84
    4  (0.4, 0.2)  {'criterion': 'entropy', 'max_depth': 50, 'max...            1.00          0.71           0.81
    5  (0.5, 0.2)  {'criterion': 'entropy', 'max_depth': 35, 'max...            1.00          0.80           0.82

### **********Performance Metric for SVM and Decision Tree***********
 Run      svm  decision_tree
   1        0.990000       0.900000
   2        0.990000       0.850000
   3        0.990000       0.840000
   4        0.990000       0.810000
   5        1.000000       0.820000
mean        0.992000       0.844000
 std        0.004472       0.035071
