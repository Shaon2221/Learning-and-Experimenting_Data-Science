#K-means clustering

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

#using elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


#Applying k-means to our dataset
kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
ymeans = kmeans.fit_predict(x)

#Visualizing our model
plt.scatter(x[ymeans==0,0],x[ymeans==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(x[ymeans==1,0],x[ymeans==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(x[ymeans==2,0],x[ymeans==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(x[ymeans==3,0],x[ymeans==3,1],s=100,c='yellow',label='Cluster 4')
plt.scatter(x[ymeans==4,0],x[ymeans==4,1],s=100,c='purple',label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='cyan',label='Centroids')
plt.title('Cluster of clients')
plt.xlabel('Annual income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()