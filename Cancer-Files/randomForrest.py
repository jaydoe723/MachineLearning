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

X_train = df1.iloc[:,2:].values
X_test = df2.iloc[:,2:].values
y_train = df3.iloc[:,1:].values.ravel()
y_test = df4.iloc[:,1:].values.ravel()

#Creates the attribute ranking table using Random forest
feat_labels = df.columns[2:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train,y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))

#Uses random forest from 1 ree to 101 trees to increase the accuracy of model
results = pd.DataFrame(columns= ['Count of Trees', 'Score for Training', 'Score for Testing'])
indexR=1
for sizeOfForest in range(1,102,10):
    forest = RandomForestClassifier(criterion='gini', n_estimators=sizeOfForest, max_depth=3)
    forest.fit(X_train, y_train)
    scoreTrain = forest.score(X_train, y_train)
    scoreTest = forest.score(X_test, y_test)
    results.loc[indexR] = [sizeOfForest, scoreTrain, scoreTest]
    indexR=indexR+1

print(results.head(16))
results.pop('Count of Trees')
results.plot()
plt.show()
