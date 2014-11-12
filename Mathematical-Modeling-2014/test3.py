#!/usr/bin/python
#coding=utf-8

import time
import numpy as np
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets.samples_generator import make_blobs

np.random.seed(0)
centers = [[1,1], [-1,-1], [1, -1]]
k = len(centers)
x , labels = make_blobs(n_samples=3000, centers=centers, cluster_std=.7)

kmeans = KMeans(init='k-means++', n_clusters=3, n_init = 10)
t0 = time.time()
kmeans.fit(x)
t_end = time.time() - t0

colors = ['r', 'b', 'g']
for k , col in zip( range(k) , colors):
    members = (kmeans.labels_ == k )
    pl.plot( x[members, 0] , x[members,1] , 'w', markerfacecolor=col, marker='.')
    pl.plot(kmeans.cluster_centers_[k,0], kmeans.cluster_centers_[k,1], 'o', markerfacecolor=col,\
            markeredgecolor='k', markersize=10)
pl.show()








