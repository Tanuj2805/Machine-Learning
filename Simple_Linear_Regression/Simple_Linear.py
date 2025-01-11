#IMPORTING LIBRARY
import numpy as np
import matplotlib as plo
import pandas as pd

#IMPORTED DATASET
dataset = pd.read_csv('Salary_Data.csv');
x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,-1].values;
print(x);
print(y);

#HANDLING IF MISSING VALUES ARE AVAILABLE
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean');
imputer.fit(x)
x = imputer.transform(x);

#DIVIDING DATA INTO TRAIN AND TEST SET
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)


#TRANING DATASET
from sklearn.linear_model import LinearRegression
regressor = LinearRegression();
regressor.fit(x_train,y_train);

#Predicting dataset
y_pred = regressor.predict(x_test)
print("Actual data set: ")
print(y_test)
print("Predicted data set")
print(y_pred)


