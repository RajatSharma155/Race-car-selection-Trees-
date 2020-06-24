# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 11:00:58 2020

@author: Rajat sharma
"""

# Importing the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the libaries
dataset = pd.read_csv('data_file.csv')
X = dataset.iloc[:, 3:-1].values
Y = dataset.iloc[:, -1].values

# Importng the catagorical data
from sklearn.preprocessing import LabelEncoder
LabelEncoder_X = LabelEncoder()
X[:, 0] = LabelEncoder_X.fit_transform(X[:, 0])
X[:, 2] = LabelEncoder_X.fit_transform(X[:, 2])
X[:, 3] = LabelEncoder_X.fit_transform(X[:, 3])

# Applying the Onehot Encoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]

# Spliting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)


#Apply the feature Scaling onto the dataset
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Applying the PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

#Fitting the K-Neighbours into the model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 60, random_state = 0, criterion= 'gini')
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred)

# K - Fold Cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train,
                             y = Y_train, cv = 10)
print(" The Accuracy {:.2f}".format(accuracies.mean() * 100))

#Visualising the Test set result
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), 
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
"""plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()].T).reshape(X1.shape),
                                       alpha = 0.75, cmap = ListedColormap('red','green')))"""
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.xlim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.show()