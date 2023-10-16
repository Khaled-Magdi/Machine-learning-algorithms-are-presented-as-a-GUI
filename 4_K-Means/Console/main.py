from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

Live= pd.read_csv('Live.csv')




#Handling Of Data
from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()
Live ['status_type'] = Le.fit_transform(Live ['status_type'])
Live ['status_published'] = pd.to_datetime(Live['status_published'])
# print(Live.status_published)

Live['status_published'] = pd.to_datetime(Live['status_published']).astype(np.int64)
# print(Live.status_published)

# print(Live['status_published'])
# print(Live.dtypes)




X = Live.iloc[:,1:].values
Y = np.random.rand(7050,0)
# print(X[:,1])





#Splitting data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=44, shuffle =True)

# print(Y_test)
# print(X_test)
#
#
# print('X_train shape is ' , X_train.shape)
# print('X_test shape is ' , X_test.shape)
# print('Y_train shape is ' , Y_train.shape)
# print('Y_test shape is ' , Y_test.shape)



KMeansModel = KMeans (n_clusters= 6 , init='random' ,random_state =33 , algorithm= 'auto')
#
print(KMeansModel.fit(X_train))

# Calculating Details
print('KMeansModel Train Score is : ' , KMeansModel.score(X_train))
print('KMeansModel Test Score is : ' , KMeansModel.score(X_test))
print('KMeansModel centers are : ' , KMeansModel.cluster_centers_)
print('KMeansModel labels are : ' , KMeansModel.labels_)
print('KMeansModel intertia is : ' , KMeansModel.inertia_)
print('KMeansModel No. of iteration is : ' , KMeansModel.n_iter_)
print('----------------------------------------------------')

# #Calculating Prediction
# Y_pred = KMeansModel.predict(X_test)
# print('Predicted Value for KMeansModel is : ' , Y_pred[:10])


# Calculating Prediction
Y_pred = KMeansModel.predict(X_test)
print('Predicted Value for KMeansModel is : ', Y_pred[:10])
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

ilist = []
n = 10
for i in range(1, n):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    ilist.append(kmeans.inertia_)

plt.plot(range(1, n), ilist)
plt.title('Elbow')
plt.xlabel('clusters')
plt.ylabel('inertias')
plt.show()



kmeans = KMeans(n_clusters=5,init='k-means++', random_state=33,algorithm= 'auto')
y_kmeans = kmeans.fit_predict(X)





# Visualising the clusters



plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=10, c='r')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=10, c='b')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=10, c='g')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=10, c='c')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=10, c='m')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=10, c='y')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='y')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()