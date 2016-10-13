import pylab as pl
import numpy as np
from scipy.spatial import distance
from sklearn.datasets.samples_generator import make_blobs


def k_means(dataset, k, max_it=1000):
    labeled_ds = np.append(dataset, np.zeros((len(dataset), 1)), axis=1)

    centers = labeled_ds[np.random.choice(len(labeled_ds), k, replace=False)]
    centers[:, -1] = range(1, k + 1)  # Assign labels

    current_it = 0
    old_centers = None

    while current_it < max_it and not np.array_equal(old_centers, centers):
        current_it += 1
        old_centers = np.copy(centers)
        update_labels(labeled_ds, centers)
        update_centers(labeled_ds, centers)

    return labeled_ds


def update_labels(dataSet, centers):
    for datum in dataSet:
        datum[-1] = centers[0, -1]
        minDist = distance.euclidean(datum[:-1], centers[0, :-1])
        for center in centers:
            dist = distance.euclidean(datum[:-1], center[:-1])
            if dist < minDist:
                minDist = dist
                datum[-1] = center[-1]


def update_centers(dataSet, centers):
    k = len(centers)
    for i in range(1, k + 1):
        cluster = dataSet[dataSet[:, -1] == i, :-1]
        centers[i - 1, :-1] = np.mean(cluster, axis=0)
        centers[i - 1, -1] = i



centers = [[-1, -1], [0, 1], [1, -1]]
X, _ = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4)
result = k_means(X, 3)

pl.figure(1)
pl.clf()
for datum in result:
        pl.plot(datum[0], datum[1], 'k.')
pl.show()

pl.figure(2)
pl.clf()
for datum in result:
    if datum[-1] == 1:
        pl.plot(datum[0], datum[1], 'r.')
    elif datum[-1] == 2:
        pl.plot(datum[0], datum[1], 'b.')
    elif datum[-1] == 3:
        pl.plot(datum[0], datum[1], 'g.')
pl.show()
