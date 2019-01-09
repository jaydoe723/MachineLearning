import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#Reading the training and testing sets and placing them in a numpy array
df = pd.read_csv('cancerNormalized.csv', sep = ',')
df1 = pd.read_csv('X_train.csv', sep=',')
df2 = pd.read_csv('X_test.csv', sep=',')
df3 = pd.read_csv('y_train.csv', sep=',')
df4 = pd.read_csv('y_test.csv', sep=',')

#Deriving an additional descriptive feature based on the radius and perimeter of the sample
df1['1+3']=df1['1']+df1['3']
df2['1+3']=df2['1']+df1['3']

X_train = df1.iloc[:,2:].values
X_test = df2.iloc[:,2:].values
y_train = df3.iloc[:,1:].values.ravel()
y_test = df4.iloc[:,1:].values.ravel()

#Apply the kNN using weighted euclidean to see if any improvement
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
