import numpy as np
import torch
from sbi.utils import BoxUniform
import matplotlib.pyplot as plt



def init_population(N0):
    shape = (2, 1)
    n = np.zeros(shape, dtype=int)
    n[:, 0] = N0 // 2
    w = np.ones(shape, dtype=float)
    u = np.zeros(shape, dtype=int)
    return n, w, u

def clean_arrays(n, w, u):
    idx = n > 0
    idx = idx[0, :] | idx[1, :]
    n = n[:, idx]
    w = w[:, idx]
    u = u[:, idx]
    return n, w, u

def step(n, w, u, Ub, tau, alpha, beta, N):
    x = n * w
    x /= x.sum()
    n = np.random.multinomial(N, x.reshape(-1)).reshape((2, -1))
    muts = np.random.binomial(n, [[tau * Ub], [Ub]])
    n -= muts
    mutants_shape = (2, muts.sum())
    nx = np.zeros(mutants_shape, dtype=int)
    wx = np.ones(mutants_shape, dtype=float)
    ux = np.zeros(mutants_shape, dtype=int)
    fitness_effects = 1 + np.random.gamma(shape=alpha, scale=beta, size=muts.sum())
    muts_non_zero = muts.nonzero()
    used_effects = 0
    mutant_j = 0
    for i, j, num_muts in zip(muts_non_zero[0], muts_non_zero[1], muts[muts_non_zero]):
        for effect in fitness_effects[used_effects:(used_effects + num_muts)]:
            nx[i, mutant_j] = 1
            wx[i, mutant_j] = w[i, j] * effect
            ux[i, mutant_j] = u[i, j] + 1
            mutant_j += 1
        used_effects += num_muts
    n = np.concatenate((n, nx), axis=1)
    w = np.concatenate((w, wx), axis=1)
    u = np.concatenate((u, ux), axis=1)
    return n, w, u

def calc_GFP(history):
    t = sorted(history.keys())
    arrays = (history[t_] for t_ in t)
    return np.array([n[0, :].sum() / n.sum() for n, w, _ in arrays])

def calc_wbar(history):
    t = sorted(history.keys())
    arrays = (history[t_] for t_ in t)
    return np.array([(n * w).sum() / n.sum() for n, w, _ in arrays])

def calc_ubar(history):
    t = sorted(history.keys())
    arrays = (history[t_] for t_ in t)
    return np.array([(n * u).sum() / n.sum() for n, _, u in arrays])

def simulator(theta, N0=100000, gens_per_bottle=5, total_gens=300, reps=1):
    Ub, alpha, beta = theta
    Ub = 10 ** Ub
    beta = 10 ** beta
    tau = 1
    results = []
    for _ in range(reps):
        history = {}
        n, w, u = init_population(N0)
        history[0] = (n, w, u)
        N = N0
        for t in range(1, total_gens + 1):
            if t % gens_per_bottle == 0:
                N = N0
            else:
                N *= 2
            n, w, u = step(n, w, u, Ub=Ub, tau=tau, alpha=alpha, beta=beta, N=N)
            n, w, u = clean_arrays(n, w, u)
            if t % 25 == 0:
                history[t] = (n, w, u)
        GFP = calc_GFP(history)
        wbar = calc_wbar(history)
        ubar = calc_ubar(history)
        result = np.concatenate([GFP, wbar, ubar])
        if wbar[-1] > 10:
            result = np.full_like(result, np.nan, dtype=np.float32)
        results.append(torch.tensor(result, dtype=torch.float32))
    return torch.stack(results)
