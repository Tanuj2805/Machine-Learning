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
imputer = SimpleImputer(missing_values=np.nan,strategy='mean') #median and most_frequenlty is also available but mean is most acceptable
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
print("Data replaced with missing value")
print(x)


#Encoding Independent Variables

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder="passthrough")
x = np.array(ct.fit_transform(x)) #Converting this to numpy array as we want a x to be numpy array while training data
print(x)

#Encoding Dependent Variables

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

#Splliting data into test and train set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

from sklearn.preprocessing import StandardScaler
fs = StandardScaler()
