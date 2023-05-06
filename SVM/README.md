# Spam mail detection using SVM

Use SVM to classify emails into spam or non-spam categories and report the classification accuracy for various SVM parameters and kernel functions.

## Dataset 

- `spambase.data` : 
    - Number of Instances: 4601 (1813 Spam = 39.4%)
    - Number of Attributes: 58 (57 continuous, 1 nominal class label)
    - Class Distribution:
        Spam	  1813  (39.4%)
        Non-Spam  2788  (60.6%)

*[More info](http://www.ics.uci.edu/~mlearn/MLRepository.html)*

## Tasks performed

1. Data pre-processing:
    
    a. Null checking
    
    b. Class symbol conversion (1,-1)
    
    c. Train-Test split (70:30)
    
    d. Normalize based on mean and variance of train split
    
2. Model building:

    a. params: kernel function (linear, poly, RBF) , soft margin constant

    b. methods:<br>
        i. `fit(X,y)` : Solves dual equation of SVM and stores weights and bias of separator<br>
        ii. `project(X)` : To project data points using obtained weights and bias<br>
        iii. `predict(X)` : Sign function to specify class label

3. Comparison with SVM defined in scikit-learn module

4. Hyper-parameter tuning

5. Visualization: Plotting decision boundary

## Results (Accuracies)

1. Our SVM results

| Kernel | C1       | C2       | C3       |
| ------ | -------- | -------- | -------- |
|        | 1        | 10       | 100      |
| linear | 0.923968 | 0.923244 | 0.919623 |
| poly   | 0.902969 | 0.897900 | 0.898624 |
| rbf    | 0.837075 | 0.849385 | 0.847212 |

2. Scikit learn results

| Kernel | C1       | C2       | C3       |
| ------ | -------- | -------- | -------- |
|        | 1        | 10       | 100      |
| linear | 0.923968 | 0.923244 | 0.918899 |
| poly   | 0.843592 | 0.924692 | 0.915279 |
| rbf    | 0.934830 | 0.934106 | 0.920348 |
