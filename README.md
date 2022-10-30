# Assignment 4

### Accuracy for SVM and Decision Tree Classifier for digit classification:

#### Best Result for SVM

| Run | Split Ratio |              Hyper parameter |   Train Accuracy |   Dev Accuracy |   Test Accuracy |
| --- | --- | --- | --- | --- | --- |
|    1 | (0.8, 0.1) | {'C': 0.3, 'gamma': 0.001}    |        0.99     |     0.99      |     0.99 |
|    2 | (0.7, 0.1) | {'C': 0.2, 'gamma': 0.001}    |        0.99     |     0.96      |     0.99 |
|    3 | (0.6, 0.1) | {'C': 0.3, 'gamma': 0.001}    |        0.99     |     0.98      |     0.99 |
|    4 | (0.4, 0.2) |   {'C': 1, 'gamma': 0.001}    |        1.00     |     0.99      |     0.99 |
|    5 | (0.5, 0.2) | {'C': 0.7, 'gamma': 0.001}    |        1.00     |     0.99      |     0.99 |

#### SVM Stats
| stats |          Run | Train Accuracy | Dev Accuracy | Test Accuracy |
| --- | --- | --- | --- | --- |
|count | 5.000000    |    5.000000   |   5.000000     |      5.00 |
|mean  | 3.000000    |    0.994000   |   0.982000     |      0.99 |
|std   | 1.581139    |    0.005477   |   0.013038     |      0.00 |
|min   | 1.000000    |    0.990000   |   0.960000     |      0.99 |
|25%   | 2.000000    |    0.990000   |   0.980000     |      0.99 |
|50%   | 3.000000    |    0.990000   |   0.990000     |      0.99 |
|75%   | 4.000000    |    1.000000   |   0.990000     |      0.99 |
|max   | 5.000000    |    1.000000   |   0.990000     |      0.99 |

#### Best Result for Decision Tree

| Run | Split Ratio |              Hyper parameter |   Train Accuracy |   Dev Accuracy |   Test Accuracy |
| --- | --- | --- | --- | --- | --- |
|    1  |(0.8, 0.1) | {'criterion': 'entropy', 'max_depth': 25, 'max_features': 'sqrt'}    |         1.0   |       0.85    |       0.87 |
|    2  |(0.7, 0.1) | {'criterion': 'entropy', 'max_depth': 40, 'max_features': 'sqrt'}    |         1.0   |       0.84    |       0.89 |
|    3  |(0.6, 0.1) | {'criterion': 'entropy', 'max_depth': 20, 'max_features': 'sqrt'}    |         1.0   |       0.86     |      0.87 |
|    4  |(0.4, 0.2) | {'criterion': 'entropy', 'max_depth': 35, 'max_features': 'auto'}    |         1.0   |       0.76     |      0.82 |
|    5  |(0.5, 0.2) | {'criterion': 'entropy', 'max_depth': 40, 'max_features': 'sqrt'}    |         1.0   |       0.81     |      0.83 |


#### Decision Tree Stats

| stats |          Run | Train Accuracy | Dev Accuracy | Test Accuracy |
| --- | --- | --- | --- | --- |
| count | 5.000000       |      5.0  |    5.000000   |    5.000000 |
| mean  | 3.000000       |      1.0  |    0.824000   |    0.856000 |
| std   | 1.581139       |      0.0  |    0.040373   |    0.029665 |
| min   | 1.000000       |      1.0  |    0.760000   |    0.820000 |
| 25%   | 2.000000       |      1.0  |    0.810000   |    0.830000 |
| 50%   | 3.000000       |      1.0  |    0.840000   |    0.870000 |
| 75%   | 4.000000       |      1.0  |    0.850000   |    0.870000 |
| max   | 5.000000       |      1.0  |    0.860000   |    0.890000 |


### Performance Metric for SVM and Decision Tree

| Run  |    svm | decision_tree |
| --- | --- | --- |
|   1 0.99   |    0.870000 |
|   2 0.99   |    0.890000 |
|   3 0.99   |    0.870000 |
|   4 0.99   |    0.820000 |
|   5 0.99   |    0.830000 |
| mean 0.99  |     0.856000 |
|  std 0.00  |     0.029665 |