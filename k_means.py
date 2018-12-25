from utils import *
import numpy as np
from copy import deepcopy
from sklearn.metrics.pairwise import euclidean_distances

def k_means(data_points, n_clusters, max_iter, eps=0.001):
    """
    K-Means Algorithm
    """
    if max_iter <= 0:
        raise ValueError('Invalid value of max iteration: %d' % max_iter)
    np.random.seed(0)
    random_indices = np.random.randint(0, data_points.shape[0], n_clusters)
    centroids = data_points[random_indices]
    clusters = None
    error = 1e9
    for i in range(max_iter):
        # When, after an update, the estimate of that center stays the same, exit loop
        if error <= eps:
            break

        # The eucliean distances to centroids
        dists = euclidean_distances(data_points, centroids)
        # Assign all training data to closest center, (for each point we chose the closest distance to a cluster)
        # clusters variable will contain the indices of the chosen cluster for each data point
        clusters = np.argmin(dists, axis=1)
#         print(clusters.shape)
#         np.set_printoptions(threshold=np.nan)
#         print(clusters)
        centroids_old = deepcopy(centroids)
        # Calculate the mean for every cluster and update the centroids.
        for cluster_index in range(n_clusters):
            cluster_points = data_points[clusters == cluster_index]
#             print(str(cluster_index) + ': ' + str(cluster_points.shape))
            if cluster_points.shape[0] > 0: # if there is at least one point that belong it this cluster
                centroids[cluster_index] = np.mean(cluster_points, axis=0)
#                 print(centroids[cluster_index])
        
#         print(centroids.shape)
        error = np.linalg.norm(centroids - centroids_old)
    return clusters, centroids
    

class KMeans():
    """ 
    """
    def __init__(self, n_clusters=DEFAULT_N_CLUSTERS, max_iter=K_MEANS_DEFAULT_MAX_ITER):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        return k_means(X, self.n_clusters, self.n_clusters)


