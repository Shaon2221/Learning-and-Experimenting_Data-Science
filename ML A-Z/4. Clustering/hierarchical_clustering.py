#hierarchical clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

#Using the dendrograms to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrograms')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

#Fitting hierarchical clustering to dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(x)

#Visualising the clusters
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c='red',label='Careful')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c='blue',label='Standard')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c='green',label='Target')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c='yellow',label='Careless')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,c='purple',label='Sensible')
plt.title('Cluster of clients')
plt.xlabel('Annual income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()