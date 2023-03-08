import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
data = 'S:\CODING NOTES\PYTHON3.0\CSV files\selling_data.csv'
df = pd.read_csv(data)
df.shape
df.head()
df.info() #View summary of dataset
df.isnull().sum() #Check for missing values in dataset
df.drop(['react_comment_r', 'react_share_r'], axis=1, inplace=True) #Drop null columns
df.info() #Again view summary of dataset
df.describe() #View the statistical summary of numerical variables
# view the labels in the variable
df['status_id'].unique()
# view how many different types of variables are there
len(df['status_id'].unique())
# view the labels in the variable
df['status_published'].unique()
# view how many different types of variables are there
len(df['status_published'].unique())
# view the labels in the variable
df['status_type'].unique()
# view how many different types of variables are there
len(df['status_type'].unique())
df.drop(['status_id', 'status_published'], axis=1, inplace=True)
df.info() #View the summary of dataset again
df.head() #Preview the dataset again
#Declare feature vector and target variable
X = df
y = df['status_type']
#Convert categorical variable into integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['status_type'] = le.fit_transform(X['status_type'])
y = le.transform(y)
#View the summary of X
X.info()
#Preview the dataset X
X.head()
#Feature Scaling
cols = X.columns
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
X = ms.fit_transform(X)
X = pd.DataFrame(X, columns=[cols])
X.head()
#K-Means model with two clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)
#Check quality of weak classification by the model
labels = kmeans.labels_
# check how many of the samples were correctly labeled
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
#Use elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
 kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
 kmeans.fit(X)
 cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()
#K-Means model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
# check how many of the samples were correctly labeled
labels = kmeans.labels_
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
#K-Means model with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)
# check how many of the samples were correctly labeled
labels = kmeans.labels_
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
#K-Means model with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X)
# check how many of the samples were correctly labeled
labels = kmeans.labels_
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))