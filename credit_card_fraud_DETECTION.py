

# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

data = pd.read_csv("/training_data.csv")

data.head()

print(data.shape)

print(data.describe())

fraud = data[data['Time'] == 1]
valid = data[data['Time'] == 0]
outlierFraction = len(fraud)/float(len(valid))
print(outlierFraction)
print('Fraud Cases: {}'.format(len(data[data['Time'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Time'] == 0])))

print ('Amount details of the fraudulent transaction')
fraud.Amount.describe()

print('details of valid transaction')
valid.Amount.describe()

# dividing the X and the Y from the dataset
X = data.drop(['Time'], axis = 1)
Y = data["Time"]
# getting just the values for the sake of processing
# (its a numpy array with no columns)
xData = X.values
yData = Y.values
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size = 0.2, random_state = 42)

# dividing the X and the Y from the dataset
X = data.drop(['Time'], axis = 1)
Y = data["Time"]
print(X.shape)
print(Y.shape)
# getting just the values for the sake of processing
# (its a numpy array with no columns)
xData = X.values
yData = Y.values
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(
        xData, yData, test_size = 0.2, random_state = 42)

# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True,cmap="RdYlGn")
plt.show()

data.isnull().values.any()

count_classes = pd.value_counts(data['Time'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Transaction Time Distribution")

plt.xticks(range(2), LABELS)

plt.xlabel("Time")

plt.ylabel("Frequency")

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by Time')
bins = 50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(valid.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();

# We Will check Do fraudulent transactions occur more often during certain time frame ? Let us find out with a visual representation.

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Transaction amount')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(valid.Time, valid.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()

import pandas as pd

# Load your datasets
train_data = pd.read_csv('/training_data.csv')
test_data = pd.read_csv('/testing_data.csv')
train_labels = pd.read_csv('/train_data_classlabels.csv')

# Check for missing values in each dataset
missing_values_train = train_data.isnull().sum()
missing_values_test = test_data.isnull().sum()
missing_values_labels = train_labels.isnull().sum()

print("Missing values in training data:\n", missing_values_train)
print("Missing values in testing data:\n", missing_values_test)
print("Missing values in training labels:\n", missing_values_labels)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Can also use 'median' or 'most_frequent'
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
test_data = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)

# Check for missing values in each dataset
missing_values_train = train_data.isnull().sum()
missing_values_test = test_data.isnull().sum()
missing_values_labels = train_labels.isnull().sum()

print("Missing values in training data:\n", missing_values_train)
print("Missing values in testing data:\n", missing_values_test)
print("Missing values in training labels:\n", missing_values_labels)

print(train_data.shape)
print(train_labels.shape)

# Trimming train_labels to match the number of samples in train_data
trimmed_train_labels = train_labels[:train_data.shape[0]]

# Verifying the new shapes
trimmed_train_labels.shape, train_data.shape

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Assuming you have defined xTrain and yTrain earlier in your code

# Initialize the RandomForestClassifier
random_forest_classifier = RandomForestClassifier()

# Now, you can fit the RandomForestClassifier
random_forest_classifier.fit(train_data,trimmed_train_labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_data,trimmed_train_labels, test_size=0.3, random_state=42)

# Predicting on the test set
y_pred_rf = random_forest_classifier.predict(X_test)

# Evaluating the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

accuracy_rf, report_rf
print('Accuracy: ',accuracy_rf)
print('Report: ',report_rf)

from sklearn.ensemble import RandomForestClassifier

# Train a RandomForestClassifier to determine feature importances
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)

importances = forest.feature_importances_

# Selecting features based on importances
indices = np.argsort(importances)[::-1]
top_k_indices = indices[:10]  # Selecting top 10 features

# Assuming X_train and X_test are pandas DataFrames
X_train_new = X_train.iloc[:, top_k_indices]
X_test_new = X_test.iloc[:, top_k_indices]

# X_train_new and X_test_new now contain only the top 10 features

"""Feature Selection (PCA-10 features)"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Assuming 'train_data' and 'trimmed_train_labels' are your dataset

# Standardize the Data
scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(train_data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_train_data, trimmed_train_labels, test_size=0.3, random_state=42)

# Apply PCA
# Let's keep 95% of the variance
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Now, X_train_pca and X_test_pca are your transformed datasets

from sklearn.metrics import accuracy_score, classification_report

# Create the RandomForestClassifier model
random_forest_classifier = RandomForestClassifier()

# Train the model with PCA-transformed data
random_forest_classifier.fit(X_train_pca, y_train)

# Predicting on the PCA-transformed test set
y_pred_rf = random_forest_classifier.predict(X_test_pca)

# Evaluating the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

print('Accuracy:', accuracy_rf)
print('Classification Report:\n', report_rf)

from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Assuming 'train_data' and 'trimmed_train_labels' are your dataset

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_data, trimmed_train_labels, test_size=0.3, random_state=42)

# Create the SVM model
svm_model = svm.SVC()

# Train the model
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred_svm = svm_model.predict(X_test)

# Evaluate the model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)

print('SVM Accuracy:', accuracy_svm)
print('Classification Report:\n', report_svm)

"""Feature Selection on SVM"""

from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report

# Create the SVM model
svm_classifier = svm.SVC()

# Train the SVM model with PCA-transformed data
svm_classifier.fit(X_train_pca, y_train)

# Predicting on the PCA-transformed test set
y_pred_svm = svm_classifier.predict(X_test_pca)

# Evaluating the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)

print('SVM Accuracy:', accuracy_svm)
print('SVM Classification Report:\n', report_svm)

"""hyperparameter tuning for rfc"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset (replace this with your own dataset)
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier()

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model from the grid search
best_rf_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_rf_model.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Set:", accuracy)

"""hyperparametric tuning for svm"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Load the iris dataset (replace this with your own dataset)
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Create a Support Vector Machine classifier
svm_classifier = SVC()

# Define the hyperparameter grid to search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model from the grid search
best_svm_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_svm_model.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Set:", accuracy)

"""k fold cross validation for random forest classifier"""

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load the iris dataset (replace this with your own dataset)
iris = load_iris()
X, y = iris.data, iris.target

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier()

# Perform K-fold cross-validation (e.g., with 5 folds)
num_folds = 5
cross_val_scores = cross_val_score(rf_classifier, X, y, cv=num_folds, scoring='accuracy')

# Print the cross-validation scores for each fold
print("Cross-validation Scores:", cross_val_scores)

# Print the average accuracy across all folds
print("Average Accuracy:", cross_val_scores.mean())

"""k fold cross validation for svm"""

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# Load the iris dataset (replace this with your own dataset)
iris = load_iris()
X, y = iris.data, iris.target

# Create a Support Vector Machine classifier
svm_classifier = SVC()

# Perform K-fold cross-validation (e.g., with 5 folds)
num_folds = 5
cross_val_scores = cross_val_score(svm_classifier, X, y, cv=num_folds, scoring='accuracy')

# Print the cross-validation scores for each fold
print("Cross-validation Scores:", cross_val_scores)

# Print the average accuracy across all folds
print("Average Accuracy:", cross_val_scores.mean())