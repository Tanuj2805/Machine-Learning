import numpy as np
import matplotlib as plo
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv');
x = dataset.iloc[:,0].values;
y = dataset.iloc[:,1].values;
print(x);
print(y);