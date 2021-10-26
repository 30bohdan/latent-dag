import os
import sys
import numpy as np
import pandas as pd
from time import time
from synthetic_data import Latent_and_Bipartite_graph, gen_samples
from ten_decomp_latent import reconstruct_ten_graph
from learn_mixture import run_full_learning, reconstruct_graph_learner, reconstruct_Ph
from pycausal.pycausal import pycausal as pyc
import itertools

def count_accuracy(B_true, B_est):
    """
    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    # SHD = undirected extra (skeleton) + undirected missing (skeleton) + reverse (directed graph)
    # unoriented_correct = # undirected edges in the cpdag that has a corresponding true edge in the true dag
    """
    d = len(B_true)
    assert (len(B_est) == d)
    undirected_extra = 0
    undirected_missing = 0
    reverse = 0
    unoriented_correct = 0
    for i in range(d):
        for j in range(i + 1, d):
            undir_true = (B_true[i][j] == 1 or B_true[j][i] == 1)
            undir_est = (B_est[i][j] == 1 or B_est[i][j] == -1 or B_est[j][i] == 1 or B_est[j][i] == -1)

            if undir_true and (not undir_est):
                undirected_missing += 1
            elif (not undir_true) and undir_est:
                undirected_extra += 1
            elif undir_true and undir_est:
                if B_est[i][j] == -1 or B_est[j][i] == -1:
                    # Undirected edge in est
                    unoriented_correct += 1
                elif B_true[i][j] != B_est[i][j]:
                    # Directed edge in est, but reversed
                    reverse += 1
    return {"shd": undirected_extra + undirected_missing + reverse,
            "undirected_extra": undirected_extra,
            "undirected_missing": undirected_missing,
            "reverse": reverse,
            "unoriented_correct": unoriented_correct}

def get_metrics(LBG, est_latent_dag_edges, est_adj, mapping):
    # mapping maps estimated vertices to true vertices
    num_hidden, num_observed = LBG.num_hidden, LBG.num_observed
    B_true = [[0 for i in range(num_hidden + num_observed)] for j in range(num_hidden + num_observed)]
    # B_true = np.zeros((num_hidden + num_observed, num_hidden + num_observed))
    for u, v in LBG.latent_dag_edges:
        B_true[u][v] = 1
    for u in range(num_hidden):
        for v in range(num_observed):
            if LBG.bip_adjacency[v, u] == 1:
                B_true[u][num_hidden + v] = 1

    B_est = [[0 for i in range(num_hidden + num_observed)] for j in range(num_hidden + num_observed)]
    # B_est = np.zeros((num_hidden + num_observed, num_hidden + num_observed))
    for u in range(num_hidden):
        for v in range(u + 1, num_hidden):
            if (u, v) in est_latent_dag_edges and (v, u) in est_latent_dag_edges:
                B_est[mapping[u]][mapping[v]] = -1
                # B_est[mapping[u], mapping[v]] = -1
            elif (u, v) in est_latent_dag_edges:
                B_est[mapping[u]][mapping[v]] = 1
                # B_est[mapping[u], mapping[v]] = 1
            elif (v, u) in est_latent_dag_edges:
                B_est[mapping[v]][mapping[u]] = 1
                # B_est[mapping[v], mapping[u]] = 1
    for u in range(num_hidden):
        for v in range(num_observed):
            if est_adj[v, u] == 1:
                B_est[mapping[u]][num_hidden + v] = 1
                # B_est[mapping[u], num_hidden + v] = 1

    metrics = count_accuracy(B_true, B_est)
    return metrics

def fges_disc_bic(df, pc):
    from pycausal import search as s
    tetrad = s.tetradrunner()
    tetrad.run(algoId = 'fges', 
               dfs = df, 
               scoreId = 'disc-bic-score', 
               dataType = 'discrete',
               verbose = False)
    dag = []
    for edge in tetrad.getEdges():
        if edge:
            u = int(edge[0])
            v = int(edge[6])
            if edge[2:5] == "---":
                dag.append((u, v))
                dag.append((v, u))
            elif edge[2:5] == "-->":
                dag.append((u, v))
            elif edge[2:5] == "<--":
                dag.append((v, u))
    return dag

def run_ges(est_prob, pc):
    probs = {}
    for idx, value in np.ndenumerate(est_prob):
        probs[idx] = value

    tuple_probs = [(tup, prob) for tup, prob in probs.items()]
    all_tuples = [tup for tup, prob in tuple_probs]
    all_probs = [prob for tup, prob in tuple_probs]
    sample_tuples = [all_tuples[i] for i in np.random.choice(list(range(len(all_tuples))), size = 100000, p = all_probs)]

    num_hidden = len(est_prob.shape)
    columns = [str(i) for i in range(num_hidden)]
    df = pd.DataFrame(sample_tuples, columns = columns)
    return columns, fges_disc_bic(df, pc)

def run_expt(outf, stats_dict, pc, num_hidden = 3, num_observed = 7, h_high = 3, 
             dim_of_var = 5, num_samples = 10000, latent_dag_density = 0.6, bip_graph_density = 0.5, distinct_dom = False):
    """
        Runs the experiments:
            outf - output file where all the training statistics is dumped
            stats_dict - dictionary where all the runtime statistics is recorded
            pc - instance of pycausal from  pycausal.pycausal
            num_hidden, num_observed  - numbers of hidden and observed variables respectively
            h_high  - the largest size of the domain (number of states) for every latent variable
            dim_of_var - dimention of every observed variable
            num_samples - number of observed samples
            latent_dag_density, bip_graph_density - edge densities for DAG latent structure 
                        and bipartite latent structure
            distinct_dom - variable enforcing all latent variables to have distinct domain sizes 

    """

    outf.write("Chosen Parameters:\n")
    outf.write(f"num_hidden = {num_hidden}, num_observed = {num_observed}, dim_of_var = {dim_of_var}\n")
    outf.write(f"latent_dag_density = {latent_dag_density}, bip_graph_density = {bip_graph_density}\n")
    outf.write(f"num_samples = {num_samples}, h_high = {h_high}\n\n")
    LBG = Latent_and_Bipartite_graph(num_hidden = num_hidden, 
                                     latent_dag_density = latent_dag_density,
                                     bip_graph_density = bip_graph_density, 
                                     num_observed = num_observed, 
                                     high = h_high,
                                     distinct_dom = distinct_dom)
    if distinct_dom and len(LBG.Hdom) == 0:
        outf.write("Cannot generate distinct domain sizes, aborting\n")
        return

    outf.write("Hdom sizes: " + str(LBG.Hdom) + '\n')
    outf.write("Latent DAG edges: " + str(LBG.latent_dag_edges) + '\n')
    outf.write("Bipartite graph adj matrix:\n")
    outf.write(np.array2string(LBG.bip_adjacency) + "\n")
    # np.savetxt(outf, LBG.bip_adjacency, fmt='%d')
    outf.write("\n")
    outf.write("True probabilities:\n")
    outf.write(str(LBG.final_probs) + "\n")

    total_comps = 1
    for x in LBG.Hdom:
        total_comps *= x
    if (total_comps > 50):
        outf.write(f"Way too many components ({total_comps}), aborting\n")
        return

    stats_dict["num_expts"] += 1

    samples, true_cluster_sizes = gen_samples(LBG, num_samples = num_samples, dim_of_var = dim_of_var)
    outf.write("Samples shape: " + str(samples.shape) + "\n")
    outf.write("True cluster sizes: " + str(true_cluster_sizes) + "\n")
    outf.write("---------------------------\n")
    outf.flush()

    max_comps_to_learn = total_comps + 5

    try:
        learner = run_full_learning(observed = num_observed, 
                                    dim = dim_of_var, 
                                    samples = samples, 
                                    comp_single_bound = max_comps_to_learn, 
                                    comp_upper_bound = max_comps_to_learn)
    except Exception as e:
        outf.write("Oracle training failed\n")
        outf.write(str(e))
        outf.write("\n")
        stats_dict["clustering_failure"] += 1
        return

    try:
        adj, Hdom = reconstruct_graph_learner(learner.return_oracle(), n_observed = num_observed, method = 'Jenrich')
        if (adj.shape[1] != num_hidden):
            outf.write("Jennrich failed, wrong number of hidden variables\n")
            stats_dict["jennrich_failure"] += 1

            outf.write("Trying ALS with hint\n")
            adj, Hdom = reconstruct_graph_learner(learner.return_oracle(), num_observed, method = 'ALS', hint = num_hidden)
    except Exception as e:
        outf.write("Tensor decomposition failed")
        outf.write(str(e))
        outf.write("\n")
        stats_dict["tensor_decomp_failure"] += 1
        return

    est_total_comps = 1
    for i in Hdom:
        est_total_comps *= i 
    if (est_total_comps > max_comps_to_learn):
        outf.write("Either oracle or Tensor decomposition failed because est_total_comps is more than max_comps_to_learn\n")
        stats_dict["too_many_est_comps"] += 1
        return

    outf.write("Recovered Hdom:\n")
    outf.write(np.array2string(Hdom) + "\n")
    
    outf.write("\n")
    outf.write("Recovered adjacency matrix:\n")
    outf.write(np.array2string(adj) + "\n")
    
    outf.write("\n")


    # Check if domain sizes match
    if sorted(list(Hdom)) != sorted(list(LBG.Hdom)):
        outf.write("Hdom sizes don't match!!\n")
        stats_dict["Hdom_mismatch"] += 1
        # return

    try:
        learner, est_prob, est_layout = reconstruct_Ph(learner, adj, Hdom)
        est_prob = est_prob / np.sum(est_prob)
    except Exception as e:
        outf.write("Ph recovery failed\n")
        outf.write(str(e))
        outf.write("\n")
        stats_dict["ph_recovery_failure"] += 1
        return

    probs = {}
    for idx, value in np.ndenumerate(est_prob):
        probs[idx] = value
    outf.write("Estimated probabilities:\n")
    outf.write(str(probs) + "\n")
    outf.write("Estimated layout:\n")
    outf.write(np.array2string(est_layout) + "\n")

    try:
        cols, latent_dag = run_ges(est_prob, pc)
    except Exception as e:
        outf.write("Learning latent DAG failed\n")
        outf.write(str(e))
        outf.write("\n")
        stats_dict["latent_dag_failure"] += 1
        return
    outf.write("Estimated Latent DAG: " + str(latent_dag) + "\n")

    stats_dict["success"] += 1

    # Metrics
    mapping = None
    if distinct_dom:
        mapping = list(range(num_hidden))
        M = {}
        for i in range(num_hidden):
            M[LBG.Hdom[i]] = i
        for i in range(num_hidden):
            mapping[i] = M[Hdom[i]]
    else:
        for perm in itertools.permutations(list(range(num_hidden))):
            match = True
            for i in range(num_hidden):
                for j in range(num_observed):
                    if LBG.bip_adjacency[j, perm[i]] != adj[j, i]:
                        match = False
                        break
            if match:
                mapping = list(perm)
                break

    # Get metrics
    final_metrics = {}
    if not mapping:
        outf.write("Bipartite graphs don't match, try all permutations and take min SHD\n")
        final_metrics = {}
        mapping = list(range(num_hidden))
        for perm in itertools.permutations(list(range(num_hidden))):
            mapping = list(perm)
            metrics = get_metrics(LBG, latent_dag, adj, mapping)
            if not final_metrics:
                final_metrics = metrics
            elif final_metrics["shd"] > metrics["shd"]:
                final_metrics = metrics
    else:
        outf.write("Bipartite graphs match, mapping is " + str(mapping) + "\n")
        final_metrics = get_metrics(LBG, latent_dag, adj, mapping)
    stats_dict["metrics_list"].append(final_metrics)

    # Dumping everything, maybe needed for drawing graphs
    expt_dump = {"num_hidden": LBG.num_hidden,
                 "num_observed": LBG.num_observed,
                 "dim_of_var": dim_of_var,
                 "true_latent_dag_edges": LBG.latent_dag_edges,
                 "true_bip_adjacency": repr(LBG.bip_adjacency),
                 "true_dom_sizes": LBG.Hdom,
                 "true_probs": LBG.final_probs,
                 "true_cluster_sizes": true_cluster_sizes,
                 "est_latent_dag_edges": latent_dag,
                 "est_bip_adjacency": repr(adj),
                 "est_dom_sizes": repr(Hdom),
                 "est_probs": probs,
                 "est_layout": repr(est_layout),
                 "metrics": final_metrics}
    stats_dict["expt_dump"].append(expt_dump)

if __name__ == '__main__':
    settings = [(1, 3, 5), 
                (2, 5, 3), (2, 5, 6), 
                (3, 7, 4), (3, 7, 5), (3, 8, 3), (3, 8, 4), (3, 8, 5),
                (4, 7, 3), (4, 8, 3)]

    #settings = [(1, 3, 5), (2, 5, 3)]

    np.printoptions(precision=5)
    print("Number of experiments per setting:", sys.argv[1])
    print("Output file:", sys.argv[2])
    print("Output file for log:", sys.argv[3])
    print("Number of samples:", sys.argv[4])
    outf = open(sys.argv[2], 'a+')
    outl = open(sys.argv[3], 'a+')

    full_stats_dict = {}

    pc = pyc()
    pc.start_vm(java_max_heap_size = '100M')
    for setting in settings:
        outf.write("Setting: " + str(setting) + "\n")
        stats_dict = {"num_expts": 0,
                      "clustering_failure": 0, 
                      "jennrich_failure" : 0,
                      "als_failure" : 0,
                      "tensor_decomp_failure" : 0,
                      "too_many_est_comps" : 0,
                      "ph_recovery_failure" : 0,
                      "latent_dag_failure": 0,
                      "Hdom_mismatch": 0,
                      "success" : 0,
                      "metrics_list" : [],
                      "expt_dump": []}

        cur_len = 0
        while (stats_dict["success"] < int(sys.argv[1])):
            start_time = time()
            num_expts_so_far = stats_dict["success"]
            outl.write(f"Experiment {num_expts_so_far + 1}/{sys.argv[1]}\n")
            outl.flush()
            outf.write(f"Experiment {num_expts_so_far + 1}/{sys.argv[1]}\n")

            num_hidden, num_observed, h_high = setting
            run_expt(outf, 
                     stats_dict, 
                     pc,
                     num_hidden = num_hidden,
                     num_observed = num_observed,
                     h_high = h_high,
                     num_samples = int(sys.argv[4]),
                     distinct_dom = False)

            if cur_len < len(stats_dict["metrics_list"]):
                outf.write("Current metrics: " + str(stats_dict["metrics_list"][-1]) + "\n")
                cur_len += 1

            outf.write("--------------------\n")
            outf.write("Dump of stats_dict so far: " + str(stats_dict) + '\n')
            outl.write("Running time of this experiment: " + str(time() - start_time) + "\n")
            outl.write("======================\n\n\n")
            outl.flush()
        outf.write("Dump of stats_dict for this setting\n")
        outf.write(str(stats_dict) + '\n')
        outf.write("======================\n\n\n\n")
        full_stats_dict[setting] = stats_dict
        outf.write("Dump of full_stats_dict\n")
        outf.write(str(full_stats_dict) + '\n')
        outf.write("=======================================\n\n\n\n\n\n\n")

    outf.write("Final dump of full_stats_dict\n")
    outf.write(str(full_stats_dict) + '\n')
    outf.close()
    pc.stop_vm()