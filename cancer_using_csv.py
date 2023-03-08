import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
df=pd.read_csvr('S:\CODING NOTES\PYTHON3.0\CSV files\iris.csv')
print(df)
print(df.dtypes)
plt.xlabel('weight')
plt.ylabel('size')
df.plot.scatter(x='weight', y='size')
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(df,test_size=0.2)
x_train=train_set.iloc[:,0:2].values
y_train=train_set.iloc[:,2].values
x_test=train_set.iloc[:,0:2].values
y_test=train_set.iloc[:,2].values
print(len(x_train))
print(len(x_test))
from sklearn.svm import SVC
model=SVC(kernel='rbf',random_state=1)
model.fit(x_train,y_train)
model.score(x_test,y_test)
model.predict([[55,4]])

