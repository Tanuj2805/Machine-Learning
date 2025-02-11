import numpy as np
import matplotlib.pyplot as plo
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,-1].values;

print(x)
print(y)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
                                             #Transformer Name as per you, Object of ecoder you want, column Number (You can Also Add more encode in tuple
tranformer = ColumnTransformer(transformers=[('Encode',OneHotEncoder(), [3])], remainder="passthrough")
x = np.array(tranformer.fit_transform(x));
print(x)

from sklearn.model_selection import train_test_split
xtrain , xtest,ytain,ytest = train_test_split(x,y,test_size=0.2,random_state=0)

print(xtrain)
print(xtest)
print(ytain)
print(ytest)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression();
regressor.fit(xtrain,ytain);

ypred = regressor.predict(xtest)
np.set_printoptions(precision=2)#Set tp print only 2 decimal points while printing float

print(ypred)
print(ytest)