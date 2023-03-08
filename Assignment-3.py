#In python create Adaboost classifier using breast cancer dataset .
# use three different be classifier and analyse the accuracy percision and recall
# import warnings filter
print("MV SUJAN KUMAR ")
print("20MIC0084")
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
cancer = datasets.load_breast_cancer()
X=cancer.data
y=cancer.target

#Using AdaBoost classifier
print("Using AdaBoost classifier:")
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
abc=AdaBoostClassifier(n_estimators=30,learning_rate=2)
model=abc.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("accuracy:",metrics.accuracy_score(y_test,y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("================================================================")

#Using SVM
print("Using SVM:")
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import metrics
svc=SVC(probability=True,kernel='linear')
abc=AdaBoostClassifier(n_estimators=50,base_estimator=svc,learning_rate=1)
model=abc.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("accuracy:",metrics.accuracy_score(y_test,y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("================================================================")

#Using RandomForestClassifier
print("Using RandomForestClassifier:")
from sklearn.ensemble import RandomForestClassifier
abc =AdaBoostClassifier(n_estimators=70, base_estimator=RandomForestClassifier(random_state = 101),learning_rate=1,random_state=96)
model=abc.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("accuracy:",metrics.accuracy_score(y_test,y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("================================================================")

#Using DecisionTreeClassifier
print("Using DecisionTreeClassifier:")
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=1)
abc =AdaBoostClassifier(n_estimators=40, base_estimator=dtree,learning_rate=0.004,random_state = 86)
model=abc.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("accuracy:",metrics.accuracy_score(y_test,y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("================================================================")