# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 10:17:16 2018

Hierarchical Clustering

@author: Arnob
"""

# import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Mall_Customers.csv')

# here we take anual income and spending (label)
X = dataset.iloc[:, [3,4]].values
 
# Dendograms to find the optimal number of clusters
# import lib for hirearchy
import scipy.cluster.hierarchy as sch

# create plt of dendrogram
Dendogram = sch.dendrogram(sch.linkage(X, method='ward')) # ward methods is try to minimize varient within each cluster 
plt.title('dendogram')
plt.xlabel('Customer')
plt.ylabel('Euclidean Distance')
plt.show()

# fitting hierarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# showing the plot
plt.plot(X, y_hc, color='blue')

# Visulization showing the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Carefull Cluster')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Standend Cluster')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Teget Cluster')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Careless Cluster')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Sencible Cluster')
#plt.scatter(hc.cluster_centers_[:, 0], 
#            hc.cluster_centers_[:, 1], 
#            s = 300, c = 'yellow', 
#            label = 'Centroids')

plt.title('Clusters of customers of HC result')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


