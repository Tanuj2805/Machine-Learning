import numpy as np
import matplotlib as plo
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv');
x = dataset.iloc[:,0].values;
y = dataset.iloc[:,1].values;
print(x);
print(y);


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)