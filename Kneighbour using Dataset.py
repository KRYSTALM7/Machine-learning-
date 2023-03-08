# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import metrics
import pandas as pd
# Loading data
df=pd.read_csv("S:\CODING NOTES\PYTHON3.0\CSV files\OwnDataSet.csv")
# Create feature and target arrays
X=df.drop("obesity",axis=1) #Feature matrix
y=df["obesity"]
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=4)
model = knn.fit(X_train, y_train)
# Predict on dataset which model has not seen before
print(knn.predict(X_test))
# Predict the response for test dataset
y_pred = model.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
