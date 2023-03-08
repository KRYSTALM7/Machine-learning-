from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
digits = load_digits()
print("Image data shape",digits.data.shape)
print("label Data Shape",digits.target.shape)
from sklearn.model_selection import train_test_split
x_train,x_test ,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.25,random_state=0)
x_train
plt.imshow(np.reshape(x_train[1],(8,8)),cmap='gray')
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train,y_train)
logisticRegr.predict(x_test[3].reshape(1,-1))               
logisticRegr.predict(x_test[0:10])
predictions = logisticRegr.predict(x_test)
predictions
score=logisticRegr.score(x_test,y_test)
print(score)
from sklearn import metrics
cm=metrics.confusion_matrix(y_test,predictions)
print(cm)


