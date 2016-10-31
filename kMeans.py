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
        old_centers = np.copy(centers)
        update_labels(labeled_ds, centers)
        update_centers(labeled_ds, centers)
        current_it += 1

    return labeled_ds


def update_labels(dataset, centers):
    for datum in dataset:
        datum[-1] = centers[0, -1]
        minDist = distance.euclidean(datum[:-1], centers[0, :-1])
        for center in centers:
            dist = distance.euclidean(datum[:-1], center[:-1])
            if dist < minDist:
                minDist = dist
                datum[-1] = center[-1]


def update_centers(dataset, centers):
    k = len(centers)
    for i in range(1, k + 1):
        cluster = dataset[dataset[:, -1] == i, :-1]
        centers[i - 1, :-1] = np.mean(cluster, axis=0)
        centers[i - 1, -1] = i


random_points, _ = make_blobs(n_samples=4000,
                              centers=[[-1, -1], [0, 1], [1, -1]],
                              cluster_std=0.5)
result = k_means(random_points, 3)

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
