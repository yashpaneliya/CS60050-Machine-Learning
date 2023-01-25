import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

df = pd.read_csv('polydipsia.csv')

y = df.Prediction.to_numpy() # The label array

features = ['Pregnancies', 'BloodPressure', 'HormoneLevel', 
            'HumulinLevel', 'BMI', 'Age', 'LineageFactor']
X = df[features].to_numpy() # The feature matrix

# Print sample dataset statistic
nsamples, nfeatures = X.shape
print("# of total samples: %s"%nsamples)
print("# of features: %s"%nfeatures)
print("Sample X:")
print(X[:5, :5])
print("Sample y:")
print(y[:5])
print()

print('Skewness of the features:')
print(df.skew())
print()

kf5 = KFold(n_splits=5, shuffle=True) # For five fold cross-validation

# Iterate over the data folds
for i, (train_indices, test_indices) in enumerate(kf5.split(X)):
    print('Fold: %s'%i)

    # Train split
    X_train = X[train_indices]
    y_train = y[train_indices]

    # Test split
    X_test = X[test_indices]
    y_test = y[test_indices]

    # Save the data into files to use those later

    gnb = GaussianNB() # The Gaussian Naive Bayes classifier
    # more at: https://scikit-learn.org/stable/modules/naive_bayes.html
    y_pred = gnb.fit(X_train, y_train).predict(X_test) # Training and then predicting using the trained model
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['no', 'yes'], zero_division=1))
