import random

import numpy as np
from scipy.spatial import distance

def kMeans(dataSet, k, maxIt):
    numPoints, numDim = dataSet.shape
    dsLabeled = np.zeros((numPoints, numDim + 1))
    dsLabeled[:, : -1] = dataSet

    centroids = dsLabeled[np.random.choice(len(dsLabeled), k, replace=False)]
    centroids[:, -1] = range(1, k + 1) # Assign labels

    currentIt = 0
    old_centroids = None

    while currentIt < maxIt and not np.array_equal(old_centroids, centroids):
        currentIt += 1
        old_centroids = np.copy(centroids)
        relabel(dsLabeled, centroids)
        centroids = get_centroids(dsLabeled, k)

    return dsLabeled

def relabel(dataSet, centroids):
    for datum in dataSet:
        datum[-1] = centroids[0, -1]
        minDist = distance.euclidean(datum[:-1], centroids[0, :-1])
        for centroid in centroids:
            dist = distance.euclidean(datum[:-1], centroid[:-1])
            if dist < minDist:
                minDist = dist
                datum[-1] = centroid[-1]

def get_centroids(dataSet, k):
    result = np.zeros((k, dataSet.shape[1]))
    for i in range(1, k + 1):
        cluster = dataSet[dataSet[:, -1] == i, :-1]
        result[i - 1, :-1] = np.mean(cluster, axis=0)
        result[i - 1, -1] = i

    return result

X = np.matrix([[2, 1, 9],
               [1, 2, 1],
               [4, 5, 1],
               [5, 4, 9]])

result = kMeans(X, 2, 10)
print(result)
