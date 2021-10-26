import numpy as np
from tensorly.decomposition import tucker, CP

# We implement tensor decomposition algorithms in this file

def jenrich_decomp(ten, eps = 1e-4):
    """ 
    Given an n x n x n tensor of the form w_1 U_1^3 + w_2 U_2^3 + ...
    Attempt to decompose it using Jennrich's algorithm.
    Returns rank, U (rows could be scaled)
    """
    n = ten.shape[0]
    a = np.random.normal(size = (n, 1))
    b = np.random.normal(size = (n, 1))
    Ta = np.dot(ten, a).reshape(n, n)
    Tb = np.dot(ten, b).reshape(n, n)
    U = np.dot(Ta, np.linalg.pinv(Tb))
    sigma, R = np.linalg.eig(U)
    sigma, R = np.real(sigma), np.real(R)
    X = R[:, np.abs(sigma)>eps]
    return X.shape[1], X


def reconstruct_ten_graph(oracle_score, n, method = 'ALS', rank = None):
    """
    Input: 
    oracle_score - oracle function that return log(number of components)
    n - number of observed variables
    method: "Jenrich" or "ALS"
    Returns 
    (1) the adjacency matrix between hidden and observed variables
    (2) Domain sizes of hidden variables
    """
    ten = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                ten[i, j, k] = oracle_score([i, j, k])

    if (method == 'Jenrich'):
        rank, U = jenrich_decomp(ten)
        # candidate adjacency matrix
        adj = (np.abs(U)>0.3/np.sqrt(n)).astype('int')
        # learn weights
        comp = np.zeros((rank, n*n*n))
        for i in range(rank):
            comp[i] = np.kron(adj[:, i], np.kron(adj[:, i], adj[:, i]))
        coef = np.linalg.lstsq(comp.T, ten.reshape(n*n*n))[0]
        Hdom = (np.exp(np.log(2)*coef))
    if (method == 'ALS'):
        if rank is None:
            est_rank, _ = jenrich_decomp(ten)
        else:
            est_rank = rank
        cp_dec = CP(est_rank, normalize_factors = True)
        sigma, components = cp_dec.fit_transform(tensor = ten) 
        components = np.asarray(components)
        adj = (components[0]+components[1]+components[2])/3
        adj = (np.abs(adj)>0.3/np.sqrt(n)).astype('int')
        sigma = sigma / (np.sum(adj, axis = 0)**(1.5))
        Hdom = (np.exp(np.log(2)*sigma))
    return adj, Hdom
