import numpy as np
import matplotlib.pyplot as plo
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,-1].values;

print(x)
print(y)

from sklearn.compose import ColumnTransformer