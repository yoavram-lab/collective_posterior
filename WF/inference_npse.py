import numpy as np
import pandas as pd
import math
# import pickle
import torch
# import seaborn as sns
import numpy as np
import sys  
import warnings
warnings.simplefilter('ignore', Warning)

import sbi
print(sbi.__version__)
from sbi.inference import NPSE
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

N = 1e7
generation = pd.read_csv('empirical_data/Chuong_116_gens.txt').columns.astype('int')

def WF(parameters, seed=None):
    """ CNV evolution simulator
    Simulates CNV and SNV evolution for x generations
    Returns proportion of the population with a CNV for specific generations
    
    Parameters
    -------------------
    N : int = 10M
        population size  
    s_snv : float
        fitness benefit of SNVs  
    m_snv : float 
        probability mutation to SNV   
    generation : np.array, 1d 
        with generations to output
    seed : int
    
    s_cnv : float
        fitness benefit of CNVs  
    m_cnv : float 
        probability mutation to CNV 
    p_0: float
        fraction of population with GAP1 CNV before beginning of experiment
    """
    # CNV parameters
    s_cnv, m_cnv, p_0 = 10**parameters

    # SNV parameters as constants
    s_snv = 1e-3
    m_snv = 1e-5
    
    if seed is not None:
        np.random.seed(seed=seed)
    else:
        np.random.seed()

    
    
    
    # Order is: wt, cnv+, cnv-, snv
    
    w = np.array([1, 1 + s_cnv, 1 + s_cnv, 1 + s_snv], dtype='float64')
    S = np.diag(w)
    
    # make transition rate array
    M = np.array([[1 - m_cnv - m_snv, 0, 0, 0],
                [m_cnv, 1, 0, 0],
                [0, 0, 1, 0],
                [m_snv, 0, 0, 1]], dtype='float64')
    assert np.allclose(M.sum(axis=0), 1)
    
    
    # mutation and selection
    E = M @ S

    # rows are genotypes, p has proportions after initial (unreported) growth
    n = np.zeros(4)
    n[2] = N*p_0 # cnv-
    n[0] = N*(1-p_0) # wt
    
    # follow proportion of the population with CNV
    # here rows will be generation, columns (there is only one) is replicate population
    p_cnv = []
    
    # run simulation to generation 116
    for t in range(int(generation.max()+1)):    
        p = n/N  # counts to frequencies
        p_cnv.append(p[1])  # frequency of reported CNVs
        p = E @ p.reshape((4, 1))  # natural selection + mutation        
        p /= p.sum()  # rescale proportions
        n = np.random.multinomial(N, np.ndarray.flatten(p)) # random genetic drift
    ret = np.transpose(p_cnv)[generation.astype(int)]
    noise = np.random.normal(0,0.02,size=ret.shape)
    return torch.tensor(ret+noise)


def wrapper(simulator, reps, parameters, seed=None):
    rep_1 = simulator(parameters)
    if simulator == WF:
        rep_1 = rep_1.reshape(1,-1)
    out_reps = torch.empty((reps, rep_1.shape[1]))
    out_reps[0,:] = rep_1
    for i in range(1,reps):
        out=simulator(parameters)
        out_reps[i,:] = out
    return out_reps


def wrapper_hierarchical(simulator, reps, parameters, var=0.02, seed=None):
    evo_reps = torch.empty(reps, len(generation))
    for i in range(reps):
        out=simulator(parameters+torch.normal(0, torch.abs(var*parameters)))
        evo_reps[i,:] = out
    return evo_reps


# Check prior, return PyTorch prior.
prior = BoxUniform(low=torch.tensor([-2, -7, -8]), high=torch.tensor([0, -2, -2]))
prior, num_parameters, prior_returns_numpy = process_prior(prior)

simulator = WF

# Check simulator, returns PyTorch simulator able to simulate batches.
simulator = process_simulator(WF, prior, prior_returns_numpy)

# Consistency check after making ready for sbi.
check_sbi_inputs(simulator, prior)

num_sims = 30_000

theta = prior.sample((num_sims,))
x = simulator(theta)
inference = NPSE(prior, sde_type="ve")

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(count_parameters(inference.score_estimator))
_ = inference.append_simulations(theta, x).train()
posterior_npse = inference.build_posterior(sample_with='sde')



lines = ['wt','ltr','ars','all']

for k in range(len(lines)):
    line = lines[k]
    Xs = pd.read_csv(f'empirical_data/{line}.csv', index_col=0) # observations
    X = torch.tensor(np.array(Xs), dtype=torch.float32)
    samples_npse = posterior_npse.set_default_x(X).sample((100,))
    torch.save(samples_npse, f'samples_npse_{line}_noisy.pt')

    
thetas = torch.tensor(pd.read_csv('tests/test_thetas.csv', index_col=0).values.astype('float'), dtype=torch.float32)


def evaluate(posterior, thetas, n_samples, h=False):
    n_set = 8
    all_samples = torch.empty(len(thetas), len(thetas[0])*n_samples)
    for i in range(len(thetas)):
        if h:
            X = wrapper_hierarchical(WF, n_set, thetas[i])
        else:
            X = wrapper(WF, n_set, thetas[i])
        samples = posterior.set_default_x(X).sample((n_samples,))
        
        all_samples[i,:] = samples.T.flatten()

        if i%10 == 1:
            print(f'{round(100*i/len(thetas),2)}%')
    return all_samples


# all_samples = evaluate(posterior_npse, thetas, 400)
# pd.DataFrame(all_samples).to_csv(f'tests/samples_WF_npse.csv')
# all_samples = evaluate(posterior_npse, thetas, 400, h=True)
# pd.DataFrame(all_samples).to_csv(f'tests/samples_WF_npse_h.csv')
