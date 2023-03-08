import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
data=pd.read_csv("S:\CODING NOTES\PYTHON3.0\CSV files\iris.csv")
print(data.sample(5))
print(data.head(5))
data.describe()
print(data)
print(data.describe())

df_norm=data[['sepal.length','sepal.width','petal.length','petal.width']].apply(lambda x: (x - x.min(()) / (x.max()-x.min())))
print(df_norm.sample(n=5))
print(df_norm.describe())
target=data[['variety']].replace(['Setosa','Versicolor','Virginica'],[0,1,2])
print(target.sample(n=5))
df=pd.concat([df_norm, target],axis=1)
print(df.sample(n=5))
train,test=train_test_split(df,test_size=0.3)
trainX=train[['sepal.length','sepal.width','petal.length','petal.width']]
# taking the training data features
print(trainY=train.variety)
# output of our training data
testX=test[['sepal.length','sepal.width','petal.length','petal.width']]
# taking test data features
print(testY=test.variety)
#output value of test data
print(trainX.head(5))
print(trainY.head(5))
print(trainX.head(5))
print(trainY.head(5))
# Solver is the weight optimizer: ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
clf=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(3,3),random_state=1)
print(clf.fit(trainX,trainY))
predictions = clf.predict(testX)
print(predictions)
print(testY.values)
from sklearn import metrics
print('The Accuracy of the Multilayer Perceptron is :',metrics.accuracy_score(prediction,testY))

