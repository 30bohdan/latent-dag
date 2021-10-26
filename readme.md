# "Learning latent causal graphs via mixture oracles" (NeurIPS 2021)

This is python code used to run experiments in the following paper by
Bohdan Kivva, Goutham Rajendran, Pradeep Ravikumar, Bryon Aragam: "Learning latent causal graphs via mixture oracles" (https://arxiv.org/abs/2106.15563) ([NeurIPS 2021](https://nips.cc/Conferences/2021/)).

If you find this code useful, please consider citing:
```
@inproceedings{latentdag2021,
    author = {Kivva, Bohdan and Rajendran, Goutham and Ravikumar, Pradeep and Aragam, Bryon},
    booktitle = {Advances in Neural Information Processing Systems},
    title = {{Learning latent causal graphs via mixture oracles}},
    year = {2021}
}
```

## Introduction

We study the problem of reconstructing a causal graphical model from data in the presence of latent variables. The main problem of interest is recovering the causal structure over the latent variables while allowing for general, potentially nonlinear dependencies. We provide an algorithm for reconstructing the full graphical model satysfying the assumptions listed in the paper. 

## Dependencies
- numpy 1.20.1
- scipy 1.6.2
- networkx 2.6
- more-itertools 8.7.0
- scikit-learn 0.24.1
- tensorly 0.6.0
- pycausal 1.2.1

## Contents
- `learn_mixture.py` Implementation of our algorithms and our implementation of a mixture oracle
- `ten_decomp_latent.py` Tensor decomposition subroutine to recover the bipartite graph between observed and latent variables
- `latent.py` Subroutines to run inclusion-exclusion computations
- `synthetic_data.py` A subroutine to generate synthetic data. Implements generation of random causal graph and generates samples from it. 
- `pipeline.py` Our implementation of an end-to-end pipeline for experiments. Runs the experiments from the paper.


## Reproducing experiments from the paper

To run experiments, use

```
python pipeline.py 1 output.txt output_log.txt 10000

```

- The 1 indicates the number of experiments to run for each of the following settings of (m, n, D): 
[(1, 3, 5), (2, 5, 3), (2, 5, 6), (3, 7, 4), (3, 7, 5), (3, 8, 3), (3, 8, 4), (3, 8, 5), (4, 7, 3), (4, 8, 3)]. 

Here, m is the number of hidden variables, n is the number of observed variables
and D is the maximum size of a domain of a hidden variable.
- The stats and other raw experiment outputs are printed in output.txt. For each experiment, look at the "Current metrics" line to see the metrics for that run.
- The logs are printed in output_log.txt
- The 10000 indicates the number of samples to use per experiment.

Running 1 experiment each for these 10 settings took approximately an hour in a Dell XPS 15.
For simplicity, to just run the experiment on 2 settings (1, 3, 5), (2, 5, 3), uncomment that line and run the same code (see lines 302-307 of pipeline.py), this will only take about 3 minutes on a Dell XPS 15. The Dell XPS 15 is the 9550 model, configured with the Intel Core i7-6700HQ processor and 16 GB RAM.

## Custom experiments

To run a custom experiment on a synthetic data call the function `run_expt` from `pipeline.py` 

Below is the example call to the function with the default parameters.

```r
run_expt(outf, stats_dict, pc, num_hidden = 3, num_observed = 7, h_high = 3, 
             dim_of_var = 5, num_samples = 10000, latent_dag_density = 0.6, bip_graph_density = 0.5, distinct_dom = False)

``` 

```
Runs an experiment with:
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

```

To run a custom experiment on a given data use 
```r

from learn_mixture import run_full_learning, reconstruct_graph_learner, reconstruct_Ph 


try:
    learner = run_full_learning(observed = num_observed, 
                                dim = dim_of_var, # dimention of every observed variable 
                                samples = samples, # samples from the distribution 
                                comp_single_bound = max_comps_to_learn, # upper bound for the num of mixture components for a single var
                                comp_upper_bound = max_comps_to_learn) # upper bound for the num of mixture components in the full mixture
except Exception as e:
    outf.write("Oracle training failed\n")
    outf.write(str(e))
```
The returned `learner` is an estimated `MixtureOracle`
```r
try:
    adj, Hdom = reconstruct_graph_learner(learner.return_oracle(), n_observed = num_observed, method = 'Jenrich')
    #if the number of hidden variables is known add the lines below
    #if (adj.shape[1] != num_hidden):
    #    outf.write("Jennrich failed, wrong number of hidden variables\n")
    #    adj, Hdom = reconstruct_graph_learner(learner.return_oracle(), num_observed, method = 'ALS', hint = num_hidden)
except Exception as e:
    outf.write("Tensor decomposition failed")
    outf.write(str(e))
```
The variable `adj` contains an estimation of the bipartite adjacency matrix between observed and hidden variables. 
The variable `Hdom` contains an estimation of the sizes of the domains of hidden variables. 
```
try:
    learner, est_prob, est_layout = reconstruct_Ph(learner, adj, Hdom)
    est_prob = est_prob / np.sum(est_prob)
except Exception as e:
    outf.write("Ph recovery failed\n")
    outf.write(str(e))

```
The variable `est_prob` contains an estimate of the joint probability distribution for hidden variables

## Feedback

The feedback is most welcome! You can add an issue or send an email to the authors.
        

