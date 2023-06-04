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

## Code walkthrough

1. Importing libraries
2. Reading dataset and preprocessing
    - Null checking
    - Class symbol conversion (1,-1)
    - Train-Test split (70:30)
    - Normalize based on mean and variance of train split
3. Model building
    - Class for `SVM` with functions for training and testing
    - params: kernel function (linear, poly, RBF) , soft margin constant
    - methods:
        - `fit(X,y)` : Solves dual equation of SVM and stores weights and bias of separator
        - `project(X)` : To project data points using obtained weights and bias
        - `predict(X)` : Sign function to specify class label
4. Model training and testing
5. Comparing with `sklearn` library function
6. Visualizing decision boundaries by performing PCA on data

## Steps to run the code

1. Install Jupyter Notebook or use Google Colab.
2. Open the file `ML_Assignment_2.ipynb` in Jupyter Notebook or Google Colab.
3. Run all the cells.

## Libraries used:

- pandas
- numpy
- matplotlib
- sklearn
- seaborn

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
