import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier

#Reading the training and testing sets and placing them in a numpy array
df1 = pd.read_csv('X_train.csv', sep=',')
df2 = pd.read_csv('X_test.csv', sep=',')
df3 = pd.read_csv('y_train.csv', sep=',')
df4 = pd.read_csv('y_test.csv', sep=',')

X_train = df1.iloc[:,2:].values
X_test = df2.iloc[:,2:].values
y_train = df3.iloc[:,1:].values.ravel()
y_test = df4.iloc[:,1:].values.ravel()

##############  Training and Testing various implementation of KNN ################

#Using Eulcidean distance
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train, y_train)

resultsKNN = pd.DataFrame(columns=['KNN', 'Score for Training', 'Score for Testing'])
for knnCount in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=knnCount, p=2, metric='minkowski')
    knn.fit(X_train, y_train)
    scoreTrain = knn.score(X_train, y_train)
    scoreTest = knn.score(X_test, y_test)
    resultsKNN.loc[knnCount] = [knnCount, scoreTrain, scoreTest]

print(resultsKNN.head(11))
resultsKNN.pop('KNN')
resultsKNN.plot()
plt.show()

#Using Manhattan distacnce
resultsKNN = pd.DataFrame(columns=['KNN', 'Score for Training', 'Score for Testing'])
for knnCount in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=knnCount, p=1, metric='minkowski')
    knn.fit(X_train, y_train)
    scoreTrain = knn.score(X_train, y_train)
    scoreTest = knn.score(X_test, y_test)
    resultsKNN.loc[knnCount] = [knnCount, scoreTrain, scoreTest]

print(resultsKNN.head(11))
resultsKNN.pop('KNN')
resultsKNN.plot()
plt.show()

#Euclidean with distance weights
resultsKNN = pd.DataFrame(columns=['KNN', 'Score for Training', 'Score for Testing'])
for knnCount in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=knnCount, p=2, metric='minkowski', weights ='distance')
    knn.fit(X_train, y_train)
    scoreTrain = knn.score(X_train, y_train)
    scoreTest = knn.score(X_test, y_test)
    resultsKNN.loc[knnCount] = [knnCount, scoreTrain, scoreTest]

print(resultsKNN.head(11))
resultsKNN.pop('KNN')
resultsKNN.plot()
plt.show()

#Manhattan with  Distance Weights
resultsKNN = pd.DataFrame(columns=['KNN', 'Score for Training', 'Score for Testing'])
for knnCount in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=knnCount, p=1, metric='minkowski', weights ='distance')
    knn.fit(X_train, y_train)
    scoreTrain = knn.score(X_train, y_train)
    scoreTest = knn.score(X_test, y_test)
    resultsKNN.loc[knnCount] = [knnCount, scoreTrain, scoreTest]

print(resultsKNN.head(11))
resultsKNN.pop('KNN')
resultsKNN.plot()
plt.show()
