import numpy as np
import matplotlib as plot
import pandas as pd

#Importing DataSet
dataset = pd.read_csv('Data.csv')

#Dividing Dependent & Independent Features
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

print(x)
print(y)

#Taking Care of Missing Values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
print("Data replaced with missing value")
print(x)