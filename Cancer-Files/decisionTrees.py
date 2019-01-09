import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

df = pd.read_csv('cancerNormalized.csv', sep = ',')

class_mapping = {label:idx for idx,label in enumerate(np.unique(df['Class']))}
df['Class'] = df['Class'].map(class_mapping)

X, y = df.iloc[:, 2:].values, df['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=0)

df1 = pd.DataFrame(data=X_train)
df3 = pd.DataFrame(data=X_test)

df2 = pd.DataFrame(data=y_test)
df4 = pd.DataFrame(data=y_train)

df1.to_csv('newX_train.csv')
df2.to_csv('newy_test.csv')
df3.to_csv('newX_test.csv')
df4.to_csv('newy_train.csv')


resultsEntropy = pd.DataFrame(columns=['LevelLimit', 'Score for Training', 'Score for Testing'])
for treeDepth in range(1,11):
    dct = DecisionTreeClassifier(criterion='entropy', max_depth = treeDepth, random_state=0)
    dct= dct.fit(X_train, y_train)
    dct.predict(X_test)
    scoreTrain = dct.score(X_train, y_train)
    scoreTest = dct.score(X_test, y_test)
    resultsEntropy.loc[treeDepth] = [treeDepth, scoreTrain, scoreTest]

print(resultsEntropy.head(11))
resultsEntropy.pop('LevelLimit')
ax = resultsEntropy.plot()
plt.show()

resultsGini = pd.DataFrame(columns=['LevelLimit', 'Score for Training', 'Score for Testing'])
for treeDepth in range(1,11):
    dct = DecisionTreeClassifier(criterion='gini', max_depth = treeDepth, random_state=0)
    dct= dct.fit(X_train, y_train)
    dct.predict(X_test)
    scoreTrain = dct.score(X_train, y_train)
    scoreTest = dct.score(X_test, y_test)
    resultsGini.loc[treeDepth] = [treeDepth, scoreTrain, scoreTest]

print(resultsGini.head(11))
resultsGini.pop('LevelLimit')
ax = resultsGini.plot()
plt.show()


dct = DecisionTreeClassifier(criterion='entropy', max_depth = 6, random_state=0)
dct= dct.fit(X_train, y_train)

dct.predict(X_test)
score = dct.score(X_test, y_test)
print(score)
score = dct.score(X_train, y_train)
print(score)

export_graphviz(dct, out_file='tree.dot')
