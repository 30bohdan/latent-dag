from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

 
from sklearn.manifold import TSNE
from time import time
from sklearn.mixture import GaussianMixture

from synthetic_data import *
from ten_decomp_latent import *
from latent import *

def masks_for_single(n, dim):
    """ A dictionary of one-hot encodings for each variable"""
    mask_dict = {}
    for i in range(n):
        mask = np.zeros(dim*n)
        mask[i*dim: (i+1)*dim].fill(1)
        mask = mask.astype('bool')
        mask_dict[i] = mask
    return mask_dict

def masks_for_pairs(n, dim):
    """ A dictionary of one-hot encodings for each pair of variables"""
    mask_dict = {}
    for i in range(n):
        for j in range(i+1, n):
            mask = np.zeros(dim*n)
            mask[i*dim: (i+1)*dim].fill(1)
            mask[j*dim: (j+1)*dim].fill(1)
            mask = mask.astype('bool')
            mask_dict[(i, j)] = mask
    return mask_dict

def masks_for_triples(n, dim):
    """ A dictionary of one-hot encodings for each triple of variables"""
    mask_dict = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                mask = np.zeros(dim*n)
                mask[i*dim: (i+1)*dim].fill(1)
                mask[j*dim: (j+1)*dim].fill(1)
                mask[k*dim: (k+1)*dim].fill(1)
                mask = mask.astype('bool')
                mask_dict[(i, j, k)] = mask
    return mask_dict

def compute_cluster_dist_matrix(clusters, m, metric = 'euclidean', mode = 'min'):
    """
    Takes a dictionary of m clusters as input (labeled from 0 to m-1)
    Returns the distance matrix between clusters
    """
    dist_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(i+1, m):
            res = cdist(clusters[i], clusters[j], metric = metric)
            if (res.size == 0):
                dij = 0
            if (mode == 'min'):
                dij = np.min(res)
            if (mode == 'mean'):
                dij = np.mean(res)
            if (mode == 'median'):
                dij = mp.median(res)
            dist_matrix[i, j] = dij
            dist_matrix[j, i] = dij
    return dist_matrix

def cluster_comp(samples, coord_mask_dict, n_clusters_dict, method = 'kmeans'):    
    """
    For each key in coord_mask_dict, and each n_clusters in n_clusters_dict,
    this will find the best clustering into that many clusters and also compute
    the means and silhouette scores.
    method should be 'kmeans' or 'agglomerative'
    """
    learned_comp = {}
    for key, mask in coord_mask_dict.items():
        learned_comp[key] = {}
        x_samples = samples[:, mask]
        for n_clusters in n_clusters_dict[key]:
            # Find clustering with n_clusters clusters
            learned_comp[key][n_clusters] = {}
            if (method == 'kmeans'):
                clusterer = KMeans(n_clusters=n_clusters, random_state=2020)
            if (method == 'agglomerative'):
                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(x_samples)

            # Compute silhouette scores and means
            silh_score = silhouette_score(x_samples, cluster_labels, sample_size = 1000)
            
            learned_comp[key][n_clusters]["score"] = silh_score
            if (method == 'kmeans'):
                learned_comp[key][n_clusters]["means"] = clusterer.cluster_centers_
            if (method == 'agglomerative'):
                for i in range(n_clusters):
                    learned_comp[key][n_clusters]["means"][i] = np.mean(x_samples[cluster_labels == i], axis = 0)
    return learned_comp

def fast_cluster_comp(samples, coord_mask_dict, n_clusters_dict, n_max_clusters = 100, means = None):
    """
    Similar to cluster_comp but this will do one cluster learning into n_max_clusters
    And then learns smaller clusters by merging clusters
    If means is given, then we use these as final cluster centers and essentially do no learning
    For merging clusters, we use Agglomerative Clustering
    """
    learned_comp = {}
    for key, mask in coord_mask_dict.items():
        learned_comp[key] = {}
        x_samples = samples[:, mask]
        print("learning components for {}".format(key))
        # Find the clustering with max # of clusters
        if (means is None) or (means[key].shape[0] == 0):
            max_clusterer = KMeans(n_clusters=n_max_clusters, random_state=2020)
            max_cluster_labels = max_clusterer.fit_predict(x_samples)
        else:
            max_clusterer = KMeans(n_clusters=means[key].shape[0], init = means[key], n_init=1, max_iter=1)
            n_max_clusters = means[key].shape[0]
            max_cluster_labels = max_clusterer.fit_predict(np.vstack((means[key], x_samples)))[n_max_clusters:]

        
        # Compute distance matrix, to be used for merging
        max_clusters = {}
        for i in range(n_max_clusters):
            max_clusters[i] = samples[max_cluster_labels == i]
        dist_matrix = compute_cluster_dist_matrix(max_clusters, n_max_clusters)

        for n_clusters in n_clusters_dict[key]:
            start_time = time()
            learned_comp[key][n_clusters] = {}
            merger =  AgglomerativeClustering(n_clusters=n_clusters, affinity = 'precomputed', linkage = 'average')
            merge_labels = merger.fit_predict(dist_matrix)
            cluster_labels = merge_labels[max_cluster_labels]

            # Compute silhouette score and means
            silh_score = silhouette_score(x_samples, cluster_labels, sample_size = 1000)
            
            learned_comp[key][n_clusters]["score"] = silh_score
            learned_comp[key][n_clusters]["means"] = np.zeros((n_clusters, x_samples.shape[1]))
            for i in range(n_clusters):
                learned_comp[key][n_clusters]["means"][i] = np.mean(x_samples[cluster_labels == i], axis = 0)
            
    return learned_comp


def vote_for_component_size(res_single, res_pairs, score = "score"):
    """
    performs voting between single components and pairs 
    based on the divisibility condition
    """
    
    for i in res_single.keys():
        for i_comp in res_single[i].keys():
            res_single[i][i_comp]["div_votes"] = res_single[i][i_comp][score]
            
    for pair in res_pairs.keys():
        for p_comp in res_pairs[pair].keys():
            res_pairs[pair][p_comp]["div_votes"] = res_pairs[pair][p_comp][score]
    
    #next, take into account votes of pairs
    for key, key_log in res_pairs.items():
        for i in key:
            for i_comp in res_single[i].keys():
                for count, key_comp in enumerate(key_log.keys()):
                    if (key_comp % i_comp == 0):
                        res_single[i][i_comp]["div_votes"] += (1/(2*count+4))*key_log[key_comp][score]
                        break #only one vote per pair
    
    #select best guess for single components
    best_single = {}
    for i in res_single.keys():
        best_single[i] = max(res_single[i].keys(), key= lambda x: res_single[i][x]["div_votes"])
    
    #now single components vote for pairs
    for key, key_log in res_pairs.items():
        for i in key:
            for count, key_comp in enumerate(key_log.keys()):
                if (key_comp % best_single[i] == 0):
                    res_pairs[pair][p_comp]["div_votes"] += (1/(count+2)**0.5)*res_single[i][best_single[i]][score]
                        #more than one vote per vertex
    
    
                        
    


def compute_mean_fit(single_means, pair_proj_means):
    """
    Computes the score indicating how well aligned are means for 
    clusters of a single variable X compared to means of a pair 
    of variables containing X
    returns:
    score measuring alignment
    score measuring "uniformity" of alignment
    a correspondence between indicies of means 
    """
    a = single_means.shape[0]
    b = pair_proj_means.shape[0]
    dist_table = np.zeros((a, b))
    for i in range(a):
        for j in range(b):
            dist_table[i, j] = np.linalg.norm(single_means[i]- pair_proj_means[j])
    #print(dist_table.T)
    best_fit_ind = np.argmin(dist_table, axis = 0) 
    best_fit = np.amin(dist_table, axis = 0)
    score = np.max(best_fit)
    (unique, counts) = np.unique(best_fit_ind, return_counts=True)
    uniform_freq_score = np.linalg.norm(counts - b/a)
    return score, uniform_freq_score, best_fit_ind



def vote_based_on_means(res_single, res_pairs):
    """
    performs voting between single components and pairs of components
    """
    #first, vote for yourself
    for i in res_single.keys():
        for i_comp in res_single[i].keys():
            res_single[i][i_comp]["mean_votes"] = 0
            res_single[i][i_comp]["freq_votes"] = 0
            
           
    for pair in res_pairs.keys():
        for p_comp in res_pairs[pair].keys():
            res_pairs[pair][p_comp]["mean_votes"] = 0
            res_pairs[pair][p_comp]["freq_votes"] = 0
    
    #next, take into account votes of pairs
    for key, key_log in res_pairs.items():
        mean_score_table = {}
        freq_score_table = {}
        
        for ind, i in enumerate(key):
            mean_score_table[i] = {}
            freq_score_table[i] = {}
            for i_comp in res_single[i].keys():                
                for count, key_comp in enumerate(key_log.keys()):
                    if (key_comp < i_comp):
                        continue
                    single_means = res_single[i][i_comp]["means"]
                    dim = res_single[i][i_comp]["means"].shape[1]
                    pair_means = key_log[key_comp]["means"][:, ind*dim : (ind+1)*dim]
                    score, freq_score, arrows = compute_mean_fit(single_means, pair_means)
                    #print(arrows)
                    mean_score_table[i][(i_comp, key_comp)] = score
                    freq_score_table[i][(i_comp, key_comp)] = freq_score
                    #res_pairs[pair][p_comp]["mean_arrows"][i][i_comp] = arrows

                    
            
            
            #compute combined score

            best_mean_fit =  dict(sorted(mean_score_table[i].items(), 
                                            key=lambda x: (10*x[1]+freq_score_table[i][x[0]]))[:6])
            
            _, high_mean = max(best_mean_fit.items(), key=lambda x: x[1])
            
            for p in best_mean_fit.keys():
                
                score_silh = res_single[i][p[0]]["score"]*res_pairs[key][p[1]]["score"]
                res_single[i][p[0]]["mean_votes"] += max([0, (1 - best_mean_fit[p])*score_silh])
                res_pairs[key][p[1]]["mean_votes"] += max([0, (1 - best_mean_fit[p])*score_silh])
                
    return 0

def compute_grid_of_means(comp1, comp2):
    """
    lifts means of the comp1 and comp2 to means of (comp1 x comp2)
    """
    n1, dim1 = comp1.shape
    n2, dim2 = comp2.shape
    res = np.zeros((n1*n2, dim1+dim2))
    for i in range(n1):
        for j in range(n2):
            res[i*n2+j, :dim1] = comp1[i]
            res[i*n2+j, dim1:] = comp2[j]
    return res

def compute_grid_of_means_tripple(dict, i, j, k):
    """
    uses means of pairs of components to estimate 
    locations of the means for triples of components
    """

    n1, dim1 = comp1.shape
    n2, dim2 = comp2.shape
    res = np.zeros((n1*n2, dim1+dim2))
    for i in range(n1):
        for j in range(n2):
            res[i*n2+j, :dim1] = comp1[i]
            res[i*n2+j, dim1:] = comp2[j]
    return res


def select_comp(learned_comp, top = 5, score = "score"):
    """
    We always assume that learned components is a dictionary 
    whose label is the tuple of coordinates "key"
    learned_comp[key] is a dictionary whose key is the number of components
    learned_comp[key][n_comp] is a dictionary that stores various information 
    about the components
    We select top n_comp based on learned_comp[key][n_comp][score]
    Returns a dictionary of selected components
    """
    selected_comp = {}
    for key, info in learned_comp.items():
        info_sorted = sorted(info.items(), key=lambda x: x[1][score],  reverse=True)
        selected_comp[key] = dict(info_sorted[:top])
    return selected_comp

def select_comp_names(learned_comp, top = 5, score = "score"):
    """
    We always assume that learned components is a dictionary 
    whose label is the tuple of coordinates "key"
    learned_comp[key] is a dictionary whose key is the number of components
    learned_comp[key][n_comp] is a dictionary that stores various information 
    about the components
    We select top n_comp based on learned_comp[key][n_comp][score]
    Returns a dictionary of selected components
    """
    selected_comp = {}
    for key, info in learned_comp.items():
        info_sorted = sorted(info.keys(), key=lambda x: info[x][score],  reverse=True)
        
        if (top>1):
            selected_comp[key] = info_sorted[:top]
        else:
            selected_comp[key] = int(info_sorted[0])
    return selected_comp

def filter(cap, list_f):
        res = list(range(2, cap+1))
        for i in list_f:
            new_res = []
            for j in res:
                if (j%i == 0):
                    new_res.append(j)
            res = sorted(new_res)
        return res


class ClusterInformation():

    def __init__(self, n_comp_dict):
        pass


class StructureLearning():

    def __init__(self, n_observed, var_dim, samples, comp_upper_bound = 30):
        """
        n_observed - Number of observed variables
        var_dim - Dimension of every variable X_i
        samples - Samples array of shape (num_samples x (var_dim x n_observed))
        comp_upper_bound - Our prior belief about the upper bound 
                    on the total number of components in the full mixture
        """
        self.n_observed = n_observed
        self.var_dim = var_dim
        self.masks_single = masks_for_single(n_observed, var_dim) 
        self.masks_pairs = masks_for_pairs(n_observed, var_dim)
        self.masks_triples = masks_for_triples(n_observed, var_dim)
        self.samples = samples
        self.comp_upper_bound = comp_upper_bound

        self.pairs_arrows = {}
        
        self.single_comp = {}
        self.pairs_comp = {}
        self.triples_comp = {}

        self.single_top = {}
        self.pairs_top = {}
        self.triples_top = {}

        self.best_single = {}
        self.best_pairs = {}
        self.best_triple = {}

        self.full_comp = {}
        self.full_arrows = {}

        self.graph = None
        self.Hdom = None

        self.graph_obj = None

    def learn_single_candidates(self, method = 'kmeans', comp_upper_bound = None):
        """
        method is in {'kmeans', 'agglomerative', 'mixed'}
        """
        #compute components to learn
        if (comp_upper_bound is None):
            comp_upper_bound = self.comp_upper_bound
        n_comp_dict = {}
        for i in self.masks_single.keys():
            n_comp_dict[i] = range(2, comp_upper_bound)
        #learn all components
        if (method == 'kmeans') or (method == 'agglomerative'):
            self.single_comp = cluster_comp(self.samples, self.masks_single, n_comp_dict, method = method)
        elif (method == 'mixed'):
            self.single_comp = fast_cluster_comp(self.samples, self.masks_single, n_comp_dict, n_max_clusters = comp_upper_bound)

    def suggest_comp_pairs(self):
        """
        Returns dictionary of suggested n_comp for every pair
        """
        suggested_pairs = {}
        cap = self.comp_upper_bound
        for i, i_info in self.single_top.items():
            cand = []
            for n_comp in range(2, cap):
                for r in i_info.keys():
                    if (n_comp % r == 0):
                        cand.append(n_comp)
                        break
     
            for j, j_info in self.single_top.items():
                if (i < j):
                    suggested_pairs[(i, j)] = []
                    for n_comp in cand:
                        for r in i_info.keys():
                            if (n_comp % r == 0):
                                suggested_pairs[(i, j)].append(n_comp)
                                break
        return suggested_pairs

    def learn_pairs_candidates(self, n_comp_dict, method = 'mixed'):
        """
        performs learning of the number of components for pairs of observed variables
        """
        if (method == 'kmeans') or (method == 'agglomerative'):
            self.pairs_comp = cluster_comp(self.samples, self.masks_pairs, n_comp_dict, method = method)
        elif (method == 'mixed'):
            self.pairs_comp = fast_cluster_comp(self.samples, self.masks_pairs, n_comp_dict)



    def select_best_lvl2(self, method = 'mixed'):
        """
        uses method from {'mixed', 'residuals', 'means', 'silh_score'}
        to select the best guess for the number of components in a mixture
        for every pair of observed variables
        'silh_score' just picks the number of components that maximizes silhouette score
        'residuals' performs "divisibility voting"
        'means' performs voting based on "means"
        'mixed' performs both means and divisibility voting
        """
        if (method == 'mixed'):
            vote_based_on_means(self.single_comp, self.pairs_comp)
            vote_for_component_size(self.single_comp, self.pairs_comp, score = "mean_votes")
            self.best_single = select_comp_names(self.single_comp, top = 1, score = "div_votes")
            self.best_pairs = select_comp_names(self.pairs_comp, top = 1, score = "div_votes")
        if (method == 'residuals'):
            vote_for_component_size(self.single_comp, self.pairs_comp)
            self.best_single = select_comp_names(self.single_comp, top = 1, score = "div_votes")
            self.best_pairs = select_comp_names(self.pairs_comp, top = 1, score = "div_votes")
        if (method == 'means'):
            vote_based_on_means(self.single_comp, self.pairs_comp)
            self.best_single = select_comp_names(self.single_comp, top = 1, score = "mean_votes")
            self.best_pairs = select_comp_names(self.pairs_comp, top = 1, score = "mean_votes") 
        elif (method == 'silh_score'):
            self.best_single = select_comp_names(self.single_comp, top = 1, score = "score")
            self.best_pairs = select_comp_names(self.pairs_comp, top = 1, score = "score")

        for key in self.pairs_comp.keys():
            n1 = self.best_single[key[0]]
            n2 = self.best_single[key[1]]
            cl_list = filter(self.comp_upper_bound-1, [n1, n2])
            if (len(cl_list) == 0):
                cl_list = filter(self.comp_upper_bound-1, [n1])
                print("Warning: something is wrong for pair: {}".format(key))
                print("best sizes are {}".format((n1, n2)))
                print(cl_list)

            self.best_pairs[key] = max(cl_list, key=lambda x: self.pairs_comp[key][x]["score"])




    def compute_mean_arrows(self, top_dict, best_top_dict, out_dict):
        """
        computes the equivalence classes of observed components 
        that correspond to states of hidden variables 
        that differ in a value of a single hidden variable
        """
        dim = self.var_dim
        for key, n_comp in best_top_dict.items():
            out_dict[key] = []
            for i in range(n_comp):
                out_dict[key].append(list(key))
            for ind, i in enumerate(key):
                single_means = self.single_comp[i][self.best_single[i]]["means"]                    
                top_means = top_dict[key][n_comp]["means"][:, ind*dim : (ind+1)*dim]
                _, _, best_fit_ind = compute_mean_fit(single_means, top_means)
                
                for jnd, j in enumerate(list(best_fit_ind)):
                    out_dict[key][jnd][ind] = j
                
            for t, j in enumerate(out_dict[key]):
                out_dict[key][t] = tuple(j)

    def compute_pair_mean_arrows(self):
        self.compute_mean_arrows(self.pairs_comp, self.best_pairs, self.pairs_arrows)
            



    def suggest_triple_means(self, key):
        """
        uses means of mixture components of pairs of variables
        to estimate the location of means of mixture components 
        for triples of variables
        """
        dim = self.var_dim
        n1 = self.best_single[key[0]]
        n2 = self.best_single[key[1]]
        n3 = self.best_single[key[2]]
        suggested_means = np.zeros((n1*n2*n3, 3*self.var_dim))
        it = 0
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    if ( ((i, j) in self.pairs_arrows[ (key[0], key[1]) ]) and 
                         ((i, k) in self.pairs_arrows[ (key[0], key[2]) ]) and
                         ((j, k) in self.pairs_arrows[ (key[1], key[2]) ]) 
                        ):
                        suggested_means[it, :dim]      = self.single_comp[key[0]][n1]["means"][i]
                        suggested_means[it, dim:2*dim] = self.single_comp[key[1]][n2]["means"][j]
                        suggested_means[it, 2*dim:]    = self.single_comp[key[2]][n3]["means"][k]
                        it += 1
        return suggested_means[:it]

    def learn_triples(self):
        """
        learns the best prediction for the number of components for triples of variables
        """
        suggested_means = {}
        n_clusters_dict = {}
        for i in range(self.n_observed):
            for j in range(i+1, self.n_observed):
                for k in range(j+1, self.n_observed):
                    n1 = self.best_single[i]
                    n2 = self.best_single[j]
                    n3 = self.best_single[k]
                    suggested_means[(i, j, k)] = self.suggest_triple_means((i, j, k))
                    if (suggested_means[(i, j, k)].shape[0] < 2):
                        print("No means were suggested for triple {}".format((i, j, k)))
                        n_clusters_dict[(i, j, k)] = list(range(self.comp_upper_bound))
                    else:
                        n_clusters_dict[(i, j, k)] = filter(suggested_means[(i, j, k)].shape[0], [n1, n2, n3])
                        if (len(n_clusters_dict[(i, j, k)]) == 0):
                            n_clusters_dict[(i, j, k)] = list(range(2, suggested_means[(i, j, k)].shape[0]+1))

        self.triples_comp = fast_cluster_comp(self.samples, self.masks_triples, n_clusters_dict, means = suggested_means)
        self.best_triple = select_comp_names(self.triples_comp, top = 1, score = "score")

    def return_oracle(self):  
        """
        returns our estimation of the weight function w, that is used to fill in 
        the order 3 tensor whose components are defined by columns of adjacency matrix
        between observed and latent variables 
        """      

        def est_count_comp(X):
            key = tuple(sorted(list(set(X))))
            if (len(key) == 1):
                return self.best_single[key[0]]
            elif (len(key) == 2):
                return self.best_pairs[key]
            elif (len(key) == 3):  
                return self.best_triple[key]
            else:
                return 0

        def est_score(X):
            return log_sum_intersection_new(log_sum_union_new, est_count_comp, X)
        return est_score


    def return_true_oracle(self):

        def score(X):
            return log_sum_intersection_new(log_sum_union_new, self.graph_obj.n_comp, X)
        return score 

    def set_best_single(self):
        for i in range(self.n_observed):
            i_n_comp = self.graph_obj.num_comp([i])
            self.best_single[i] = i_n_comp



    def train_full_mixture(self, method = 'kmeans'):
         """
        uses method from {'mixed', 'residuals', 'means', 'silh_score'}
        to select the best guess for the number of components in the full mixture
        """
        full_key = tuple(range(self.n_observed))

        total_n_comp = 1
        self.Hdom = np.around(self.Hdom).astype('int')
        for i in self.Hdom:
            total_n_comp = total_n_comp*i
        print(total_n_comp)

        if (method == 'kmeans'):
            clusterer = KMeans(n_clusters=total_n_comp)
        if (method == 'agglomerative'):
            clusterer = AgglomerativeClustering(n_clusters=total_n_comp)

        cluster_labels = clusterer.fit_predict(self.samples)
        silh_score = silhouette_score(self.samples, cluster_labels)
        print("silhuette for full mixture {}".format(silh_score))
        self.full_comp[full_key] = {}
        self.full_comp[full_key][total_n_comp] = {}
        if (method == 'kmeans'):
            self.full_comp[full_key][total_n_comp]["means"] = clusterer.cluster_centers_
        if (method == 'agglomerative'):
            self.full_comp[full_key][total_n_comp]["means"] = np.mean(self.samples[cluster_labels == i], axis = 0)
        labels, freq = np.unique(cluster_labels, return_counts = True)
        self.full_comp[full_key][total_n_comp]["freq"] = freq

        

        self.compute_mean_arrows(self.full_comp, {full_key: total_n_comp}, self.full_arrows)

def fill_table(table_edges, freq, Hdom, table_freq, layout, hid_completed = 0):
    """
    implements the recursive algorithm for recovering the joint probability table of hidden variables
    assumes that the table is filled for all vectors of hamming weight less than hid_completed
    reconstructs the values of the tables with entries of hamming weight = hid_completed 
    """
    n_hidden = Hdom.shape[0]
    if (hid_completed >= n_hidden):
        return 0
    elif (hid_completed == 0):
        corner = tuple([0]*n_hidden)
        table_freq[corner] = freq[0]
        layout[corner] = 0
        for h in range(n_hidden):
            for comp_ind, comp in enumerate(table_edges[0][h]):
                ind = list(corner)
                ind[h] = comp_ind+1
                ind = tuple(ind)
                table_freq[ind] = freq[comp]
                layout[ind] = comp
        
        hid_completed = 1
        fill_table(table_edges, freq, Hdom, table_freq, layout, hid_completed = hid_completed)
        return 0
    else:
        corner = tuple([0]*hid_completed)
        ind_slice = np.ndindex(tuple(Hdom[:hid_completed]))
        rest = tuple([0]*(n_hidden-hid_completed))
        for ind in ind_slice:
            if (ind == corner):
                continue
            #print("I am working with ind {}{}".format(ind, rest))
            comp_ind = layout[ind][rest]
            list_comp = table_edges[comp_ind][hid_completed]
            #print("Horizontal list is {}".format(list_comp))
            for h in range(hid_completed, n_hidden):
                for h_ind in range(Hdom[h]):
                    v_ind = list(rest)
                    v_ind[h - hid_completed] = h_ind
                    v_ind = tuple(v_ind)
                    ind0 = list(ind)
                    change_ind = hid_completed-1
                    for ci in range(hid_completed):
                        if (ind0[ci] > 0):
                            change_ind = ci
                            break
                    ind0[change_ind]  = 0
                    ind0 = tuple(ind0)
                    
                    v_list_comp = table_edges[ layout[ind0][v_ind] ][change_ind]
                    
                    for comp in list_comp:
                        if (comp in v_list_comp):
                            layout[ind][v_ind] = comp
                            table_freq[ind][v_ind] = freq[comp]
                            break

        hid_completed += 1
        fill_table(table_edges, freq, Hdom, table_freq, layout, hid_completed = hid_completed)
        return 0



def learn_structure(full_arrows, freq_list, graph, Hdom): 
    """
    reconstructs the joint probability table PH
    """
    n_observed = graph.shape[0]
    n_hidden = graph.shape[1]
    n_comp = freq_list.shape[0]
    est_prob = np.zeros(tuple(Hdom))
    est_layout = np.zeros(tuple(Hdom)).astype('int')

    full_arrows_arr = np.zeros((n_comp, n_observed), dtype = 'int')
    table_edges = {}
    for i in range(n_comp):
        full_arrows_arr[i] = np.asarray(list(full_arrows[i]), dtype = 'int')
        table_edges[i] = {}

    for h in range(n_hidden):
        not_connected = []
        for i in range(n_observed):
            if (graph[i, h] == 0):
                not_connected.append(i)
        for ni in range(n_comp): 
            table_edges[ni][h] = []
            keyi = full_arrows_arr[ni, not_connected]
            for nj in range(n_comp):
                keyj = full_arrows_arr[nj, not_connected]
                if ((ni != nj) and (np.linalg.norm(keyi-keyj)<0.2)): 
                        #numbers are ineteger, so its either 0, or >=1.
                    table_edges[ni][h].append(nj)
    print(table_edges)
    fill_table(table_edges, freq_list, Hdom, est_prob, est_layout)
    return est_prob, est_layout


def run_full_learning(observed, dim, samples, comp_single_bound = 20, comp_upper_bound = 20, G = None):
    """
    This is the first phase of our pipeline. It takes samples as an input and 
    output the predicted number of numbers of components for sets of observed variables of size at most 3
    """

    learner = StructureLearning(observed, dim, samples, comp_upper_bound = comp_upper_bound)
    learner.learn_single_candidates(method = 'mixed', comp_upper_bound = comp_single_bound)
    learner.single_top = select_comp(learner.single_comp, top=4)
    sugg_pairs = learner.suggest_comp_pairs()
    learner.learn_pairs_candidates(sugg_pairs)
    learner.select_best_lvl2(method = 'mixed')
    print("Our best guess now:")
    print(learner.best_single)
    print(learner.best_pairs)
    learner.compute_pair_mean_arrows()
    learner.learn_triples()
    if not (G is None):
            for i in learner.best_triple.keys():
                print("For var {},  True n_comp {:3d}, naive n_comp, fixed n_comp {:3d}".format(i, G.num_comp(i), \
                                                                                learner.best_triple[i]))
            for i in learner.best_pairs.keys():
                print("For var {},  True n_comp {:3d}, naive n_comp, fixed n_comp {:3d}".format(i, G.num_comp(i), \
                                                                                learner.best_pairs[i]))
            for i in learner.best_single.keys():
                print("For var {},  True n_comp {:3d}, naive n_comp, fixed n_comp {:3d}".format(i, G.num_comp([i]), \
                                                                                learner.best_single[i]))
    return learner

def reconstruct_graph_learner(oracle, n_observed, method = 'ALS', hint = None):
    """
    This is the second phase of our algorithm. It takes the function "oracle" that computes Wsne as an input
    perform tensor decomposition and outputs our prediction for the bipartite graph between
    latent variables and observed variables 
    """
    #oracle = learner.oracle
    if (method == 'Jenrich'):
        res_adj, res_Hdom = reconstruct_ten_graph(oracle, n_observed, method = 'Jenrich')
    else:
        res_adj, res_Hdom = reconstruct_ten_graph(oracle, n_observed, method = 'ALS', rank = hint)
    print("Our estimated graph and domain sizes:")
    print(res_adj)
    res_Hdom_int = np.around(res_Hdom).astype('int')
    print(res_Hdom_int)
    return res_adj, res_Hdom_int

def reconstruct_Ph(learner, adj, Hdom):
    """
    This is combined third and fourth phases of our algorithm
    It performs learning of the map L and then recovers 
    the distribution of the latent variables Ph
    """
    learner.graph, learner.Hdom = (adj, Hdom)
    learner.graph_obj = BipGraph(adj.shape[1], learner.n_observed)
    learner.graph_obj.set_dom(Hdom)
    learner.graph_obj.set_adj(adj)
    learner.set_best_single()
    learner.train_full_mixture()
    full_key = tuple(range(learner.n_observed))
    full_n_comp = list(learner.full_comp[full_key].keys())[0]

    est_prob, est_layout = learn_structure(learner.full_arrows[full_key], 
                                       learner.full_comp[full_key][full_n_comp]["freq"], 
                                       learner.graph, learner.Hdom)
    print(est_prob)
    return learner, est_prob, est_layout












                







