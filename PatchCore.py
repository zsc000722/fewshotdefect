import numpy as np
from sklearn.metrics import pairwise_distances


class Patch_Core:
    def __init__(self, X, metric='euclidean', cluster_center=None):
        self.X = X
        self.min_distances = None
        self.metric = metric
        self.cluster_center = cluster_center
        if cluster_center is not None:
            dist = pairwise_distances(self.X, self.cluster_center, metric=self.metric)
            self.min_distances = np.min(dist, axis=1).reshape(-1, 1)

    def update_distance(self, x):
        # x = self.cluster_center[-1]
        dist = pairwise_distances(self.X, x, metric=self.metric)
        self.min_distances = np.minimum(self.min_distances, dist)

    def select_data(self, num):
        new_batch = []
        print("insert nodes\t", num)
        for _ in range(num):
            temp_min_dist = self.min_distances.copy()
            temp_min_dist = np.sort(temp_min_dist)
            half = len(temp_min_dist) // 2
            mid = temp_min_dist[half]
            ind = np.where(abs(self.min_distances - mid) < 1e-5)[0][0]
            # ind = np.argmax(self.min_distances)
            new_vec = self.X[ind:ind + 1]
            # self.cluster_center = np.concatenate((self.cluster_center, self.X[ind:ind + 1]))
            self.X = np.delete(self.X, ind, axis=0)
            self.min_distances = np.delete(self.min_distances, ind, axis=0)

            self.update_distance(new_vec)
            new_batch.append(new_vec)
        return new_batch
