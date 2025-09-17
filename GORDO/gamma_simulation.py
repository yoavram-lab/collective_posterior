import os
from datetime import datetime
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
import numba

from numpy.random import binomial, multinomial, gamma


def init_population(N0):
    shape = (2, 1)
    n = np.zeros(shape, dtype=int) # counts per class, n[i,j] is jth class of ith strain
    n[:, 0] = N0//2
    w = np.ones(shape, dtype=float)  # fitness per class
    u = np.zeros(shape, dtype=int) # number of mutations per class
    return n, w, u


def clean_arrays(n, w, u):
    idx = n > 0
    idx = idx[0,:] | idx[1,:]
    n = n[:, idx]
    w = w[:, idx]
    u = u[:, idx]
    return n, w, u


def step(n, w, u, Ub, τ, α, β, N):    
    # selection
    x = n * w
    x /= x.sum()
    assert np.isclose(x.sum(), 1)
    
    # drift
    n = multinomial(N, x.reshape(-1)).reshape((2,-1))

    # mutation
    ## number of new mutants per class
    muts = binomial(n, [[τ*Ub], [Ub]])
    n -= muts # remove from original class
    assert n.sum() == N - muts.sum()

    ## create mutants
    mutants_shape = (2, muts.sum())
    nx = np.zeros(mutants_shape, dtype=int)
    wx = np.ones(mutants_shape, dtype=float)
    ux = np.zeros(mutants_shape, dtype=int)
    fitness_effects = 1 + gamma(shape=α, scale=β, size=muts.sum())
    
    muts_non_zero = muts.nonzero()
    used_effects = 0
    mutant_j = 0
    for i, j, num_muts in zip(muts_non_zero[0], muts_non_zero[1], muts[muts_non_zero]):
        for effect in fitness_effects[used_effects:(used_effects+num_muts)]:        
            nx[i, mutant_j] = 1
            wx[i, mutant_j] = w[i,j] * effect
            ux[i, mutant_j] = u[i,j] + 1
            mutant_j += 1
        used_effects += num_muts
    assert nx.sum() == muts.sum()
    ## add mutants to population
    n = np.concatenate((n, nx), axis=1)
    w = np.concatenate((w, wx), axis=1)
    u = np.concatenate((u, ux), axis=1)
    assert n.sum() == N, (n.sum(), N)

    return n, w, u


def calc_GFP(history):
    t = sorted(history.keys())
    arrays = (history[t_] for t_ in t)
    return np.array([
        n[0,:].sum() / n.sum()
        for n, w, _ in arrays
    ])
def calc_wbar(history):
    t = sorted(history.keys())
    arrays = (history[t_] for t_ in t)
    return np.array([
        (n * w).sum() / n.sum()
        for n, w, _ in arrays
    ])
def calc_ubar(history):
    t = sorted(history.keys())
    arrays = (history[t_] for t_ in t)
    return np.array([
        (n * u).sum() / n.sum()
        for n, _, u in arrays
    ])
    
def main():
    print("Starting simulation...")
    N0 = 100000
    gens_per_bottle = 5
    Nmax = N0*2**gens_per_bottle
    total_gens = 300
    reps = 100
    Ub = 10**-4.5 # beneficial mutation rate per strain from Moura De Sousa et al 2013 Fig 2
    τ = 1 # increase in mutation rate in YFP
    α = 1 # DFE Gamma parameter from Moura De Sousa et al 2013 Fig 2
    β = 10**(-1.5) # DFE Gamma parameter from Moura De Sousa et al 2013 Fig 2

    params = dict(
        N0=N0, 
        gens_per_bottle=gens_per_bottle,
        total_gens=total_gens,
        reps=reps,
        Ub=Ub,
        τ=τ,
        α=α,
        β=β
    )
    
    Ubs = np.array([[Ub], [τ * Ub]])
    N = N0
    
    histories = []
    for _ in tqdm(range(reps), 'Replications'):
        history = {}
        n, w, u = init_population(N0)
        history[0] = (n, w, u)
        N = N0
        for t in range(1, total_gens+1): 
            if t % gens_per_bottle == 0:
                N = N0
            else:
                N *= 2
            n, w, u = step(n, w, u, Ub=Ub, τ=τ, α=α, β=β, N=N)
            n, w, u = clean_arrays(n, w, u)
            if t % 25 == 0:
                history[t] = (n, w, u)
        histories.append(history)

    t = np.array(sorted(history.keys()))
    GFP = np.array([calc_GFP(history) for history in histories])
    wbar = np.array([calc_wbar(history) for history in histories])
    ubar = np.array([calc_ubar(history) for history in histories])
    
    summary = pd.DataFrame([
            dict(
                Population=pop,
                Generation=gen,
                GFP=GFP[pop, gen],
                wbar=wbar[pop, gen],
                ubar=ubar[pop, gen],
            )
            for pop in range(wbar.shape[0])
            for gen in range(wbar.shape[1])
        ],
        columns=['Population', 'Generation', 'GFP', 'wbar', 'ubar']
    )
    now = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    fname = os.path.join('output', now)
    with open(fname + '.json', 'wt') as f:
        json.dump(params, f)
    summary.to_csv(fname + '.csv', index=False)
    print("Wrote parameters to {} and output summary to {}".format(
        fname + '.json', fname + '.csv'
    ))
    
if __name__ == '__main__':
    main()