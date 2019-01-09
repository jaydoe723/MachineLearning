import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series

#open and read the file into a pandas DataFrame
df = pd.read_csv('cancerData.csv', sep = ',')

#Map the data based on the 'Class' attribute
class_mapping = {label:idx for idx,label in enumerate(np.unique(df['Class']))}
print(class_mapping)
df['Class'] = df['Class'].map(class_mapping)
print(df.head())

#Normalize to (0,2) range using Min-Max Normalization, excluding class attribute
df.iloc[:, df.columns != 'Class'] = (df- df.min()) / (df.max() - df.min())*2

#Saves normalized df to csv
df.to_csv('cancerNormalized.csv')

#Shows that the distribution for all attributes is between 0 and 2
df.pop('Class')
df.hist(grid=False, figsize=[15,10],sharex=True)
plt.show()
