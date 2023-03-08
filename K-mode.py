import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
bank = pd.read_csv('S:\CODING NOTES\PYTHON3.0\CSV files\bankmarketing.csv')
print(bank.head())
print(bank.columns)
# Importing Categorical Columns
bank_cust = bank(['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
 'previous', 'poutcome'],
 dtype='object')
bank_cust.head()
# Converting age into categorical variable.
bank_cust['age_bin'] = pd.cut(bank_cust['age'], [0, 20, 30, 40, 50, 60, 70, 80, 90, 100],
labels=['0-20', '20-30', '30-40', '40-50','50-60','60-70','70-80', '80-90','90-100'])
bank_cust = bank_cust.drop('age',axis = 1)
bank_cust.head()
bank_cust.shape
bank_cust.describe()
bank_cust.info()
# Checking Null values
bank_cust.isnull().sum()*100/bank_cust.shape[0]
# There are no NULL values in the dataset, hence it is clean.

# First we will keep a copy of data
bank_cust_copy = bank_cust.copy()

#Data Preparation
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
bank_cust = bank_cust.apply(le.fit_transform)
bank_cust.head()
# Importing Libraries
from kmodes.kmodes import KModes

#Using K-Mode with "Cao" initialization
km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(bank_cust)

# Predicted Clusters
fitClusters_cao

clusterCentroidsDf = pd.DataFrame(km_cao.cluster_centroids_)
clusterCentroidsDf.columns = bank_cust.columns

# Mode of the clusters
clusterCentroidsDf
#Using K-Mode with "Huang" initialization
km_huang = KModes(n_clusters=2, init = "Huang", n_init = 1, verbose=1)
fitClusters_huang = km_huang.fit_predict(bank_cust)

# Predicted clusters
fitClusters_huang
#Choosing K by comparing Cost against each K
cost = []
for num_clusters in list(range(1,5)):
 kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
 kmode.fit_predict(bank_cust)
y = np.array([i for i in range(1,5,1)])
plt.plot(y,cost)
## Choosing K=2
km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(bank_cust)
fitClusters_cao

#Combining the predicted clusters with the original DF
bank_cust = bank_cust_copy.reset_index()
clustersDf = pd.DataFrame(fitClusters_cao)
clustersDf.columns = ['cluster_predicted']
combinedDf = pd.concat([bank_cust, clustersDf], axis = 1).reset_index()
combinedDf = combinedDf.drop(['index', 'level_0'], axis = 1)


## Choosing K=2
km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(bank_cust)

print(combinedDf.head())