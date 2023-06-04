# Polydipsia Prediction

## Problem statement

The model will take the various diagnostic results (like hormone level, blood pressure etc.) and patient's conditions (like age, BMI, lineage factor) as input features and predict whether she has polydipsia. The model will be trained using **Gaussian Naive Bayes Classifier** algorithm. 

## Dataset

Dataset contains 768 rows and 8 columns. Dataset can be found [here](/Gaussian%20Naive%20Bayes/polydipsia.csv)

Features are:

`'Pregnancies', 'BloodPressure', 'HormoneLevel', HumulinLevel', 'BMI', 'Age', 'LineageFactor', 'Prediction'`

## Code walkthrough

1. Importing libraries
2. Reading dataset
3. Data analysis
    - Checking for null values
    - Correlation matrix
    - Distribution of features
4. Model building
    - Class for `Gaussian Naive Bayes Classifier` with functions for training and testing
5. k-fold validation
6. Training and testing of model
7. Confusion matrix and score calculation
8. Feature transformation using various functions
    - skewness identificaion
    - log(1+x) transformation
    - Quadratic transformation
9. Training and testing of model on transformed data
10. Comparing with `sklearn` library function

## Steps to run the code

1. Install Jupyter Notebook or use Google Colab.
2. Open the file `Gaussian Naive Bayes.ipynb` in Jupyter Notebook or Google Colab.
3. Run all the cells to perform data analysis, cleaning, model building and testing.

## Libraries used

1. Pandas
2. Numpy
3. Matplotlib
4. Seaborn
5. Scikit-learn (To compare our model with library function)
6. math

*Detailed report can be found [here](/Gaussian%20Naive%20Bayes/Polydipsia%20Prediction%20using%20Gaussian%20Naive%20Bayes%20Classifier%20Learning%20Model%20-%20Report.pdf)*
