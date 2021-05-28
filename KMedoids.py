from scipy.sparse import csr_matrix
import numpy as np
import random
from utils import *
from auto_tqdm import tqdm


class KMedoids:
    def __init__(self, n_cluster=2, max_iter=10, tol=0.1, start_prob=0.8, end_prob=0.99):
        '''Kmedoids constructor called'''
        if start_prob < 0 or start_prob >= 1 or end_prob < 0 or end_prob >= 1 or start_prob > end_prob:
            raise ValueError('Invalid input')
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.tol = tol
        self.start_prob = start_prob
        self.end_prob = end_prob
        
        self.medoids = []
        self.clusters = {}
        self.tol_reached = float('inf')
        self.current_distance = 0
        
        self.X = None
        self.is_csr = None
        self.rows = 0
        self.columns = 0
        self.cluster_distances = {}
        
        
    def fit(self, data, c_0=0):
        self.c_0 = c_0
        self.X = data
        self.rows = len(self.X)
        self.columns = len(self.X[0])     
        self.start_algo()
        return self
    
    def start_algo(self):
        self.medoids = list(kmeanspp(self.X, self.c_0, self.n_cluster - self.c_0))
        self.clusters, self.cluster_distances = self.calculate_clusters(self.medoids)
        self.update_clusters()
 
    def update_clusters(self):
        for i in tqdm(range(self.max_iter)):
            cluster_dist_with_new_medoids = self.swap_and_recalculate_clusters()
            if self.is_new_cluster_dist_small(cluster_dist_with_new_medoids) == True:
                self.clusters, self.cluster_distances = self.calculate_clusters(self.medoids)
            else:
                break

    def is_new_cluster_dist_small(self, cluster_dist_with_new_medoids):
        existance_dist = np.mean(list(self.cluster_distances.values()))
        new_dist = np.mean(list(cluster_dist_with_new_medoids.values()))
        print(existance_dist, new_dist)
        if existance_dist > new_dist and (existance_dist - new_dist) / existance_dist > self.tol:
            self.medoids = list(np.sort(list(cluster_dist_with_new_medoids.keys())))
            return True
        return False
        
    def swap_and_recalculate_clusters(self):
        # http://www.math.le.ac.uk/people/ag153/homepage/KmeansKmedoids/Kmeans_Kmedoids.html
        cluster_dist = {}
        for i, medoid in enumerate(self.medoids):
            if i <= self.c_0:
                cluster_dist[medoid] = self.cluster_distances[medoid]
                continue
            cluster_list = list([medoid] + self.clusters[medoid])
            pw_dists = sq_pairwise_distance(
                self.X[np.array(cluster_list)], self.X[np.array(cluster_list)])
            avg_dists = np.mean(pw_dists, -1)
            best_distance, best_medoid = np.min(avg_dists), cluster_list[np.argmin(avg_dists)]
            cluster_dist[best_medoid] = best_distance
#             if medoid != best_medoid:
#                 print(medoid, best_medoid)
        return cluster_dist
        
    def calculate_clusters(self, medoids):
        clusters = {}
        cluster_distances = {}
        medoid_distances = sq_pairwise_distance(self.X, self.X[np.array(medoids)])
        distances = np.min(medoid_distances, -1)
        assignements = np.argmin(medoid_distances, -1)
        for i, medoid in enumerate(medoids):
            idxs = np.where(assignements == i)[0]
            cluster_distances[medoid] = np.mean(distances[idxs])
            clusters[medoid] = list(idxs)
        return clusters, cluster_distances
                           
    def initialize_medoids(self):
        '''Kmeans++ initialisation'''
        if self.c_0:
            self.medoids = [i for i in range(self.c_0)]
        else:
            self.c_0 = 0
            self.medoids.append(random.randint(0, self.rows-1))
        medoids = np.array(self.medoids)
        non_medoids = np.delete(np.arange(self.rows), medoids)
        distances = np.min(sq_pairwise_distance(self.X[non_medoids], self.X[medoids]), -1)
        for i in tqdm(range(self.n_cluster - len(self.medoids))):
            new_index = weighted_choice(distances)
            medoids = np.concatenate([medoids, non_medoids[new_index: new_index+1]])
            non_medoids = np.delete(non_medoids, new_index)
            new_medoid_distances = sq_pairwise_distance(
                self.X[non_medoids], self.X[medoids[-1]: medoids[-1]+1])[:, 0]
            new_distances = np.delete(distances, new_index)
            distances = np.minimum(new_distances, new_medoid_distances)
        self.medoids = list(medoids)
        
