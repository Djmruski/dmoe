import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def cluster_diff_alg(X, class_size):
    default_base = {'quantile': .3,
                'eps': .3,
                'metric': 'cosine',
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': max(2, int(class_size/2)),
                'min_samples': 20,
                'xi': 0.05,
                'min_cluster_size': 0.1}
    params = default_base.copy()
#     params.update(algo_params)
    
    kmeans = KMeans(n_clusters=params['n_clusters'], random_state=0)
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'], random_state=0)
    # ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)
    spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack', affinity="nearest_neighbors", random_state=0)
    dbscan = cluster.DBSCAN(eps=params['eps'], min_samples=3, metric=params['metric'])
    # optics = cluster.OPTICS(min_samples=params['min_samples'], xi=params['xi'], min_cluster_size=params['min_cluster_size'])
    affinity_propagation = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'], random_state=0)
    # average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full', random_state=0)

    clustering_algorithms = (
        ('MiniBatchKMeans', two_means),
        ('KMeans', kmeans),
        ('AffinityPropagation', affinity_propagation),
        ('SpectralClustering', spectral),
        # ('Ward', ward),
        # ('AgglomerativeClustering', average_linkage),
        ('DBSCAN', dbscan),
        # ('OPTICS', optics),
        ('Birch', birch),
        ('GaussianMixture', gmm)
    )
    max_sil = -1
    cluster_labels = None
    n_clusters = 0
    if (class_size <=2):
        clusterlabel = kmeans.fit_predict(X)
        n_clusters = len(set((kmeans.labels_)))
        return n_clusters, clusterlabel
    else:
        for name, algorithm in clustering_algorithms:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                    "connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                    " may not work as expected.",
                    category=UserWarning)
                clusterlabel = algorithm.fit_predict(X)
                silhouette_avg = silhouette_score(X, clusterlabel)
                if name == 'GaussianMixture':
                    print(f"{name} silhoutte_score: {silhouette_avg:.4f}\t{algorithm.n_components}")
                else:
                    print(f"{name} silhoutte_score: {silhouette_avg:.4f}\t{len(set((algorithm.labels_)))}")
                
                # if (silhouette_avg > max_sil):
                #     max_sil = silhouette_avg
                #     cluster_labels = clusterlabel
                #     n_clusters = len(set((algorithm.labels_)))
    return n_clusters, cluster_labels

def clusterKMEANS(X, class_size):
    previous = -1 
    max_diff = 0
    range_n_clusters = list(range(2, class_size))
    previous_clusterer_labels = None
    previous_n_clusters = 2
    for n_clusters in range_n_clusters:

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
#         print("cluster labels: ", cluster_labels)
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        if previous > 0:
            diff = silhouette_avg - previous
            if diff < 0:
                print('diff < 0,', previous_n_clusters)
                return previous_n_clusters, previous_clusterer_labels
            else: 
                if max_diff > 0:
                    if diff < max_diff:
                        print('diff < max_diff,', previous_n_clusters)
                        return previous_n_clusters, previous_clusterer_labels 
                else:
                    max_diff = diff
                    print( "diff: ", diff, ", max diff:", max_diff)

        previous = silhouette_avg
        previous_clusterer_labels = cluster_labels
        previous_n_clusters = n_clusters