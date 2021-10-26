import networkx as nx
import numpy as np
import itertools
from itertools import chain, combinations
from math import log2


# This file contains a function that generates the random causal graph
# and contains the implementation of our algorithm for computing
# maximal neighborhood blocks needed in our algorithm for recovering 
# the bipaprtite structure of the causal graph (Section B)  



class randomData():
    def __init__(self, num_hidden, num_observed, prob = 0.5):
        """
         Creates a bipartite graph with hidden and observed variables 
        and each edge has probability prob of occuring
        """
        self.num_hidden = num_hidden
        self.num_observed = num_observed
        self.hidden_dom_size = np.zeros(num_hidden)
        G = nx.DiGraph()
        G.add_nodes_from([i for i in range(num_hidden)], type="hidden")
        G.add_nodes_from([i for i in range(num_hidden, num_hidden + num_observed)], type="observed")
        for i in range(G.number_of_nodes()):
            if (G.nodes[i]['type'] == "hidden"):
                G.nodes[i]['size'] = np.random.randint(2, 5)
        

        # Generate random edges
        self.adj = np.zeros((self.num_observed, self.num_hidden))

        while True:
            hid = -1
            for i in range(G.number_of_nodes()):
                for j in range(i + 1, G.number_of_nodes()):
                    obs = 0
                    if (G.nodes[i]['type'] == "hidden" and G.nodes[j]['type'] == "observed"):
                        if (obs == 0):
                            hid+=1
                        if (np.random.binomial(1, prob) == 1):
                            G.add_edge(i, j)
                            adj[obs, hid] = 1
                        obs += 1

            def good(G):
            # Check if two latent variables have the same children
                for i in range(G.number_of_nodes()):
                    for j in range(i + 1, G.number_of_nodes()):
                        if (G.nodes[i]['type'] == "hidden" and G.nodes[j]['type'] == "hidden"):
                            if (sorted(list(G.successors(i))) == sorted(list(G.successors(j)))):
                                return False
                return True
            if (good(G)):
                break
        self.G = G
        
        


    def adjacency(self):
        """ returns the |X| x |H| adjacency matrix"""
        return self.adj

    def num_comp(self, Y):
        """ 
            Returns the number of components for the subset Y
            Here, we just multiply the sizes of the parents
        """
        processed = set()
        ret = 1
        for i in Y:
            for h in self.G.predecessors(i):
                if h not in processed:
                    ret *= self.G.nodes[h]['size']
                    processed.add(h)
        return ret

    

    def observed_vars(self):
        return [i for i in range(self.num_hidden, self.num_hidden + self.num_observed)]

    def true_blocks(self):
        """
        Returns the true blocks from the generating data
        """
        true_blocks = []
        for i in range(self.G.number_of_nodes()):
            if (self.G.nodes[i]['type'] == "hidden"):
                cur_block = sorted(list(self.G.successors(i)))
                if cur_block:
                    true_blocks.append((cur_block, self.G.nodes[i]["size"]))
        return sorted(true_blocks)

    def is_equal(self, blocks):
        """
        checks if DAGs are the same
        """
        blocks = sorted([(sorted(children), size) for children, size in blocks])
        return blocks == self.true_blocks()

def log_sum_union(X, Y, exclude = None):
    """
     This is W_G(Y) from the writeup, we just ask the oracle for this
     exclude is set of blocks we don't wanna consider.
    """
    ret = log2(X.num_comp(Y))
    if exclude:
        for (S, log_size) in exclude:
            if Y.issubset(S):
                ret -= log_size
    return ret

def log_sum_intersection(X, Y, exclude = None):
    """
     This is Wsne_G(Y) from the writeup computed using inclusion-exclusion.
     exclude is set of blocks we don't wanna consider.
    """
    ret = 0
    for i in range(1, len(Y) + 1):
        for S in combinations(Y, i):
            S = list(S)
            if (i % 2 == 0):
                ret -= log_sum_union(X, S)
            else:
                ret += log_sum_union(X, S)
    if exclude:
        for (S, log_size) in exclude:
            if Y.issubset(S):
                ret -= log_size
    return ret

def log_sum_union_new(num_comp, Y, exclude = None):
    """
     This is log_sum_union with access to an oracle
    """
    ret = log2(num_comp(Y))
    if exclude:
        for (S, log_size) in exclude:
            if Y.issubset(S):
                ret -= log_size
    return ret

def log_sum_intersection_new(log_sum, num_comp, Y, exclude = None):
    """
        This is log_sum_intersection with access to an oracle
    """
    ret = 0
    for i in range(1, len(Y) + 1):
        for S in combinations(Y, i):
            S = list(S)
            if (i % 2 == 0):
                ret -= log_sum(num_comp, S)
            else:
                ret += log_sum(num_comp, S)
    if exclude:
        for (S, log_size) in exclude:
            if Y.issubset(S):
                ret -= log_size
    return ret

def get_blocks(X, Y, exclude = None):
    """
     Finds maximal neighbourhood blocks (see Def. B.5) in the subset Y. 
     In this code, Y is always the full set
     exclude contains all the maximal blocks already found
    """
    if not exclude:
        exclude = []
    for i in range(len(Y), 0, -1):
        for S in combinations(Y, i):
            # Go over all subsets S of size i
            S = set(S)
            cur_block = log_sum_intersection(X, S, exclude)
            if (cur_block >  0.5):
                # S is a maximal block, save it and recurse
                exclude.append((set(S), cur_block))
                return get_blocks(X, Y, exclude)
    # No more maximal blocks, return what we have found so far
    return [(sorted(list(children)), round(2 ** log_size)) for children, log_size in exclude]

def find_bipartite(X):
    blocks = get_blocks(X, set(X.observed_vars()))
    return blocks

if __name__ == "__main__":
    np.random.seed(123)
    num_experiments = 100
    for i in range(num_experiments):
        num_hidden = np.random.randint(2, 6)
        num_observed = np.random.randint(6, 12)
        prob = np.random.uniform(0.33, 0.66)
        print(i, num_hidden, num_observed, prob)

        # Generate random data
        X = randomData(num_hidden, num_observed, prob)

        # Run the bruteforce algorithm
        blocks = find_bipartite(X)

        # Check if same DAG was returned
        print(blocks, X.true_blocks())
        assert X.is_equal(blocks)
        print("")