import sys
import numpy as np
import networkx as nx
import itertools
import random
import collections

class LatentDAG():
    def __init__(self, num_hidden, Hdom, edge_prob = 0.5):
        self.num_hidden = num_hidden
        self.edge_prob = edge_prob
        self.Hdom = Hdom

    def gen_latent_dag_and_probs(self):
        """
        Generates a random DAG and a latent distribution
        """
        perm = np.arange(self.num_hidden)
        np.random.shuffle(perm) # topsort
        pars = [[] for i in range(self.num_hidden)]
        for i in range(self.num_hidden):
            for j in range(i + 1, self.num_hidden):
                if np.random.binomial(1, p = self.edge_prob) == 1:
                    pars[perm[j]].append(perm[i])
        

        dag_edges = []
        for i in range(self.num_hidden):
            for p in pars[i]:
                dag_edges.append((p, i))

        domain_vals = [list(range(self.Hdom[i])) for i in range(self.num_hidden)]
        all_tuples = list(itertools.product(*domain_vals))

        cond_probs = {}
        for i in range(self.num_hidden):
            parents = pars[i]
            parents_domain_vals = [list(range(self.Hdom[p])) for p in parents]
            all_parent_tuples = list(itertools.product(*parents_domain_vals))
            for tup in all_parent_tuples:
                # choose fresh randomness
                prob = []
                for t in range(self.Hdom[i]):
                    prob.append(np.random.randint(1, 5))
                prob = np.array(prob)
                prob = prob / np.sum(prob)
                cond_probs[(i, tup)] = prob
        

        final_probs = {}
        for tup in all_tuples:
            cur_prob = 1
            for i in range(self.num_hidden):
                parent_vals = []
                for p in pars[i]:
                    parent_vals.append(tup[p])
                cur_prob *= cond_probs[(i, tuple(parent_vals))][tup[i]]
            final_probs[tup] = cur_prob
        
        return dag_edges, final_probs

class BipGraph():
    """
    The class that contains infomation about the bipartite causal graph between 
    observed and hidden variables
    """
    def __init__(self, num_hidden, num_observed):
        self.num_hidden = num_hidden
        self.num_observed = num_observed
        self.hidden_dom_size = np.zeros(num_hidden)
        self.adj = np.zeros((self.num_observed, self.num_hidden))

    def gen_random_graph(self, prob = 0.5, attempts = 1000, regime = 'both'):
        """
         Creates a bipartite graph with hidden and observed variables 
         and each edge has probability prob of occuring
         has regimes {'lin-indep': ensures that the columns of the 
                                   bipartite graph are lineraly inependent 
                       'subset': ensures that the bipartite graph 
                                 satisfies SSC subset condition
                       both': ensures that both of these assumptions are satisfied}
        """
        at = 0

        if (self.num_observed <self.num_hidden) and (
            (regime == 'lin-indep') or (regime == 'both')):
            print('request is invalid')
            return 0

        while (at<attempts):
            at += 1
            self.adj = np.random.binomial(1, prob, size = self.adj.shape)
            def good(adj, regime):
                # Check if two latent variables have the same children
                # Also check that no hidden variable or observed variable have degree 0
                for i in range(self.num_hidden):
                    if (np.sum((adj[:, i])**2)<0.1):
                            return False
                    for j in range(i+1, self.num_hidden):
                        if (np.sum((adj[:, i] - adj[:, j])**2)<0.1):
                            return False
                for j in range(self.num_observed):
                    if (np.sum((adj[j, :])**2)<0.1):
                            return False

                if (regime == 'lin-indep') or (regime == 'both'):
                    if (np.linalg.matrix_rank(adj)<adj.shape[1]):
                        return False
                if (regime == 'subset') or (regime == 'both'):
                    for i in range(self.num_hidden):
                        for j in range(self.num_hidden):
                            if (i != j):
                                mask = adj[:, i].astype('bool')
                                col = adj[mask, j]
                                if (np.sum(col) == col.shape[0]):
                                    return False

                return True

            if (good(self.adj, regime)):
                break

    def gen_random_dom(self, low=2, high=4):
        """
        generates random domain sizes for hidden variables in the inclusive range [low, high]
        """
        self.hidden_dom_size = np.random.randint(low, high+1, self.num_hidden)

    def set_dom(self, Hdom):
        """
        sets the domain sizes for hidden variables to be Hdom
        """
        Hdom = np.array(Hdom)
        assert self.hidden_dom_size.shape == Hdom.shape, "Error: provided matrix has a wrong shape"
        self.hidden_dom_size = Hdom

    def set_adj(self, adj):
        """
        sets the adjacency matrix of the bipartite graph "self" to be adj
        """
        assert (self.adj.shape == adj.shape), "Error: provided matrix has a wrong shape"
        self.adj = adj

    def adjacency(self):
        """
        returns the adjacency matrix of a bipartite graph
        """
        return self.adj

    def num_comp(self, Y):
        # Return the number of components for the subset Y
        # by just multiplying the sizes of the parents
        processed = set()
        ret = 1
        for i in Y:
            for h in range(self.num_hidden):
                if (h not in processed) and (self.adj[i, h]):
                    ret *= self.hidden_dom_size[h]
                    processed.add(h)
        return ret


class Latent_and_Bipartite_graph():
    """
    Class that combines information about Latent and Bipartite causal graphs
    """
    def __init__(self, num_hidden, num_observed, latent_dag_density = 0.5, bip_graph_density = 0.5, high = 3, distinct_dom = False):
        # np.random.seed(123)
        self.num_hidden = num_hidden
        self.num_observed = num_observed
        self.Hdom = [np.random.randint(2, high + 1) for i in range(num_hidden)]
        if distinct_dom:
            allowed = list(range(2, high + 1))
            if (len(allowed) < num_hidden):
                self.Hdom = []
                return
            all_subsets = list(map(set, itertools.combinations(allowed, num_hidden)))
            self.Hdom = list(all_subsets[np.random.randint(len(all_subsets))])
            random.shuffle(self.Hdom)

        self.latentdag = LatentDAG(num_hidden, self.Hdom, edge_prob = latent_dag_density)
        self.latent_dag_edges, self.final_probs = self.latentdag.gen_latent_dag_and_probs()

        self.bipgraph = BipGraph(num_hidden, num_observed)
        self.bipgraph.set_dom(self.Hdom)
        self.bipgraph.gen_random_graph(prob = bip_graph_density)
        self.bip_adjacency = self.bipgraph.adjacency()

    def num_comp(self, Y):
        return self.bipgraph.num_comp(Y)

def tuple_to_int(tup, Hdom):
    """
    encodes a tuple into an int with respect to dimensions Hdom 
    """
    assert (len(tup) == len(Hdom)), "tuple has wrong number of elements"
    if (len(tup) == 0):
        return 1
    n = len(tup)
    code = tup[0]
    mod = Hdom[0]
    for i in range(1, n):
        code = code+mod*tup[i]
        mod = mod*Hdom[i]
    return code

def int_to_tuple(code, Hdom):
    """
    decodes an int into a tuple with respect to dimensions Hdom 
    """
    if (len(Hdom) == 0):
        return ()
    tup = [0]*len(Hdom)
    n = len(Hdom)
    for i in range(n):
        tup[i] = code % Hdom[i]
        code = code//Hdom[i]
    return tuple(tup)

def normalize(v):
    """
    normalize the vector
    """
    return v/(np.sum(v**2))**0.5

def generate_components(graph, Ph, Hdom, num_samples, dim_of_var = 5, sigma = 0.01, eps = 0.002):
    """
    generates the samples from a random distribution that has prescribed 
    graph and distribution over the hidden variables
    Input: 
        graph - (object of class randomData)
                .adjacency() return the adjacency matrix of graph
                .num_observed stores the number of observed var
                .num_hidden stores the number of hidden var
                .num_comp(S) the oracle that returns the number of mixture
                    components observed over the set S of obs var 
        Ph - the probability distribution over the hidden variables
        Hdom  == Ph.shape
        num_samples - the number of samples to be returned
        dim_of_var - dimension of each observed variable
    Output:
        returns (G_comp, samples)
        G_comp contains a dictionary with the discription 
            of Gaussian components in every coordinate
        samples contains an array of num_samples samples from G_comp 
    """
    assert (Ph.shape == tuple(Hdom)), "input is not consistent"
    assert (np.abs(np.sum(Ph) - 1)<1e-6), "Ph is not a distribution"
    obs = graph.num_observed
    hid = graph.num_hidden

    adj = graph.adjacency()

    score = np.zeros(obs, dtype = 'int')
    for i, ind in zip(range(graph.num_observed), range(obs)):
        score[ind] =  int(graph.num_comp([i]))
    G_comp = {}
    for i in range(obs):
        for j in range(score[i]):
            G_comp[(i, j, "mean")] = normalize(np.random.randn(dim_of_var))
            A = normalize(np.random.randn(dim_of_var, dim_of_var))
            G_comp[(i, j, "var")] = sigma*np.diag(normalize(np.random.randn(dim_of_var))**2)+ \
                         eps*np.dot(A, A.T)

    dom_size = 1
    for i in Hdom:
        dom_size = dom_size*i 
    
    Ph_flat = Ph.reshape(dom_size, order = 'F')
    
    sample_hid = np.random.choice(dom_size, size = num_samples, p = Ph_flat)
    
    
    samples = np.zeros((num_samples, obs*dim_of_var))
    for sh, ind in zip(sample_hid, range(num_samples)):
        
        #sh - int code of value of h (tuple)

        tuph = int_to_tuple(sh, Hdom)
        # recover the tuple
        
        x = np.zeros(obs*dim_of_var)
        for i in range(obs):
            # find the hidden variables that influence this tuple
            idt = tuple(np.asarray(tuph)[adj[i].astype('bool')])
            
            # compute the corresponding size of their domain
            iHdom = tuple(np.asarray(Hdom)[adj[i].astype('bool')])

            #recover its index
            idx = tuple_to_int(idt, iHdom)
            
            x[i*dim_of_var:(i+1)*dim_of_var] = \
                    np.random.multivariate_normal(G_comp[(i, idx, "mean")], G_comp[(i, idx, "var")])
        samples[ind] = x
    return (G_comp, samples)

def gen_samples(LBG, num_samples = 8000, dim_of_var = 5, sigma = 0.01, eps = 0.002):
    # pre-generate gaussians
    G_comp = {}
    for i in range(LBG.num_observed):
        for j in range(int(LBG.num_comp([i]))):
            G_comp[(i, j, "mean")] = normalize(np.random.randn(dim_of_var))
            A = normalize(np.random.randn(dim_of_var, dim_of_var))
            G_comp[(i, j, "var")] = sigma*np.diag(normalize(np.random.randn(dim_of_var))**2) + \
                         eps*np.dot(A, A.T)
      

    tuple_probs = [(tup, prob) for tup, prob in LBG.final_probs.items()]
    all_tuples = [tup for tup, prob in tuple_probs]
    all_probs = [prob for tup, prob in tuple_probs]
    sample_tuples = [all_tuples[i] for i in np.random.choice(list(range(len(all_tuples))), size = num_samples, p = all_probs)]
    cluster_sizes = sorted([v for k, v in dict(collections.Counter(sample_tuples)).items()])

    samples = np.zeros((num_samples, LBG.num_observed * dim_of_var))
    for samp in range(num_samples):
        for i in range(LBG.num_observed):
            parent_vals, parents_domain_sizes = [], []
            for p in range(LBG.num_hidden):
                if LBG.bip_adjacency[i, p] == 1:
                    parent_vals.append(sample_tuples[samp][p])
                    parents_domain_sizes.append(LBG.Hdom[p])

            idx = tuple_to_int(parent_vals, parents_domain_sizes)
            samples[samp][i*dim_of_var : (i+1)*dim_of_var] = \
                    np.random.multivariate_normal(G_comp[(i, idx, "mean")], G_comp[(i, idx, "var")])
    return samples, cluster_sizes

# LBG = Latent_and_Bipartite_graph(3, 7)
# print("num_hidden, num_observed:", LBG.num_hidden, LBG.num_observed)
# print("Hdom sizes:", LBG.Hdom)
# print("Latent DAG edges:", LBG.latent_dag_edges)
# print("Bipartite graph adj matrix:")
# print(LBG.bip_adjacency)
# print("==================\n\n\n")

# samples = gen_samples(LBG, num_samples = 100)
# print(samples.shape)