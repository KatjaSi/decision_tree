from random import sample, randint
from re import M
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from math import floor
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, K = 2):
        self.K = K
        self.centroids = list()
    
    def k_means_loop(self, X):
        z = assign_dp_to_centroids2(X, self.centroids)
        K = len(self.centroids)
        for i in range(K):
            x_i = X[z == i]
            self.centroids[i] = np.mean(x_i, axis = 0)  
        z = assign_dp_to_centroids2(X, self.centroids)
        return z
            

    def fit(self, X, iterations = 10, visualize = False):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        # normalize X to compensate for the difference in order of magnitude amoung coordinates
        X_p, v = normalize(X)
        self.v = v

        indx = randint(0, len(X))
        self.centroids.append(X_p.loc[indx]) # initialized first centroid at random

        five_proc = floor(len(X)*0.05) # 5 % op all datapoitns
        for _ in range(self.K-1):
            indxs = sample(range(0,len(X)), five_proc)
            # find the index of the point which is not to close to any of the already found centroids amoung the 5 % randomly chosen
            ind = max(indxs, key = lambda c: np.min([euclidean_distance(self.centroids[j], X_p.loc[c]) for j in range(len(self.centroids))]))
            self.centroids.append(X_p.loc[ind])
        
        if visualize:
                    z = assign_dp_to_centroids2(X_p, self.centroids)
                    self.visualize_km(X, z)
        z = self.k_means_loop(X_p)
        # assign data points to centroids
        for _ in range(iterations):
            z_new = self.k_means_loop(X_p)
            if (z_new == z).all():
                break
            z = z_new
            if visualize:
                    self.visualize_km(X, z)

    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        return assign_dp_to_centroids2(normalize(X)[0], self.centroids)

    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        centroids = list()
        for c in self.centroids:
            centroids.append(c * self.v)
        return  np.asarray(centroids)
    
    
    def visualize_km(self, X, z):
        C = self.get_centroids()
        K2 = len(C)
        _, ax = plt.subplots(figsize=(5, 5), dpi=100)
        sns.scatterplot(x='x0', y='x1', hue=z, hue_order=range(K2), palette='tab10', data=X, ax=ax)
        sns.scatterplot(x=C[:,0], y=C[:,1], hue=range(K2), palette='tab10', marker='*', s=250, edgecolor='black', ax=ax)
        ax.legend().remove()
    
# --- Some utility functions 

def assign_dp_to_centroids2(X, centroids):
    """
    Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
    """
    m = np.zeros(X.shape[0], dtype=int)

    indices = np.arange(len(centroids), dtype=int)
    for i in range(len(X)):
        x = X.loc[i]
        # find closest centroid
        centroid_index = min(indices, key = lambda c: euclidean_distance(x, centroids[c]))
        # assign data_point to the closest centroid
        m[i] = centroid_index
    return m

def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)



def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points 
    
    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.
            
    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))


def normalize(X):
    X_p = X.copy()
    norm_vector = list()
    for col in X.columns:
        m = np.max(np.abs(X_p[col]))
        X_p[col] = X_p[col]/m
        norm_vector. append(m)
    return X_p, np.array(norm_vector)