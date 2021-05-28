import numpy as np
import random
from auto_tqdm import tqdm
from utils import *

def greedy_k_center(X, c_0, b=1):
    "Coreset greedy algorithm."
    centers = np.arange(c_0)
    N = len(X)
    candidates = np.arange(c_0, N)
    distances = np.min(sq_pairwise_distance(X[candidates], X[centers]), -1)
    for query in tqdm(range(b)):
        new_idx = np.argmax(distances)
        centers = np.concatenate([centers, candidates[new_idx: new_idx+1]])
        candidates = np.delete(candidates, new_idx)
        new_distances = np.delete(distances, new_idx)
        new_center_distances = sq_pairwise_distance(X[candidates], X[centers[-1]: centers[-1]+1])[:, 0]
        distances = np.minimum(new_distances, new_center_distances)
    return np.max(distances), centers


def kmeanspp(X, c_0=None, b=1):
    
    N = len(X)
    if c_0:
        centers = np.arange(c_0)
    else:
        c_0 = 1
        b -= 1
        centers = np.array([0])
    candidates = np.arange(c_0, N)
    distances = np.min(sq_pairwise_distance(X[candidates], X[centers]), -1)
    for query in tqdm(range(b)):
        new_idx = np.random.choice(len(distances), p=distances/np.sum(distances))
        centers = np.concatenate([centers, candidates[new_idx: new_idx+1]])
        candidates = np.delete(candidates, new_idx)
        new_distances = np.delete(distances, new_idx)
        new_center_distances = sq_pairwise_distance(X[candidates], X[centers[-1]: centers[-1]+1])[:, 0]
        distances = np.minimum(new_distances, new_center_distances)
    return centers


def run_active_selection(data, budget, method):
    
    Lbl_x, Lbl_y = data['Lbl_x'], data['Lbl_y']
    Unl_p, Unl_x = data['Unl_p'], data['Unl_x']
    if method == 'coresetgreedy':
        _, c = greedy_k_center(
            np.concatenate([Lbl_x, Unl_x]),
            len(Lbl_x),
            budget)
        chosen = c[len(Lbl_x):] - len(Lbl_x)
    elif method == 'kmedoids':
        km = KMedoids(len(Lbl_x) + budget, max_iter=10, tol=0.1)
        km.fit(np.concatenate([Lbl_x, Unl_x]), len(Lbl_x))
        medoids = np.sort(np.array(list(km.clusters.keys())))
        chosen = medoids[len(Lbl_x):] - len(Lbl_x)
    elif method == 'badge':
        y_hat = np.argmax(Unl_p, -1)
        coef = -Unl_p
        coef[np.arange(len(coef)), y_hat] += 1
        emb = np.concatenate([Unl_x * coef[:, i:i+1] for i in range(Unl_p.shape[-1])], -1)
        chosen = kmeanspp(emb, 0, budget)
    elif method == 'minconf':
        predicted_proba = np.max(Unl_p, -1)
        chosen = np.argsort(predicted_proba)[:budget]
    elif method == 'entropy':
        Unl_p_clipped = np.clip(Unl_p, 1e-12, None)
        entropies = -np.sum(np.log(Unl_p_clipped) * Unl_p_clipped, -1)
        chosen = np.argsort(-entropies)[:budget]
    elif method == 'margin':
        Unl_p_first = np.sort(Unl_p, -1)[:, -1]
        Unl_p_second = np.sort(Unl_p, -1)[:, -2]
        margins = Unl_p_first - Unl_p_second
        chosen = np.argsort(margins)[:budget]
    else:
        raise ValueError('Invalid args.method!')            
    return chosen
        
class KMedoids(object):
    
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

