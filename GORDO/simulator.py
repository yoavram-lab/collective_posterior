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

# def simulator(theta, N0=100000, gens_per_bottle=5, total_gens=300, reps=1):
#     Ub, alpha, beta = theta
#     Ub = 10 ** Ub
#     beta = 10 ** beta
#     tau = 1
#     results = []
#     for _ in range(reps):
#         history = {}
#         n, w, u = init_population(N0)
#         history[0] = (n, w, u)
#         N = N0
#         for t in range(1, total_gens + 1):
#             if t % gens_per_bottle == 0:
#                 N = N0
#             else:
#                 N *= 2
#             n, w, u = step(n, w, u, Ub=Ub, tau=tau, alpha=alpha, beta=beta, N=N)
#             n, w, u = clean_arrays(n, w, u)
#             if t % 25 == 0:
#                 history[t] = (n, w, u)
#         GFP = calc_GFP(history)
#         wbar = calc_wbar(history)
#         ubar = calc_ubar(history)
#         result = np.concatenate([GFP, wbar, ubar])
#         # if wbar[-1] > 10:
#         #     result = np.full_like(result, np.nan, dtype=np.float32)
#         results.append(torch.tensor(result, dtype=torch.float32))
#         interim_res = torch.stack(results)
#     return interim_res


def split_outputs(batch_tensor):
    """
    Your simulator returns a torch.tensor of shape (reps, 3*T),
    concatenating GFP, wbar, ubar across T saved timepoints.
    This splits it back into three numpy arrays shaped (reps, T).
    """
    X = batch_tensor.detach().cpu().numpy()
    reps, L = X.shape
    assert L % 3 == 0, "Expected [GFP | wbar | ubar] concatenation"
    T = L // 3
    GFP  = X[:, 0:T]
    wbar = X[:, T:2*T]
    ubar = X[:, 2*T:3*T]
    return GFP, wbar, ubar


import numpy as np
import torch

# --- helpers for summary stats ---

def _ensure_reps_by_time(A: np.ndarray) -> np.ndarray:
    """Ensure A is (reps, T); transpose if it looks like (T, reps)."""
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("Expected a 2D array")
    r, t = A.shape
    return A if r >= t else A.T

def sum_freq_data_from_GFP(GFP: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """
    Histogram of |0.5 - GFP| in [0, 0.5], across replicates per time point.
    Returns (T, n_bins) with row-wise probabilities.
    """
    GFP = _ensure_reps_by_time(GFP)        # (reps, T)
    dev = np.abs(0.5 - GFP)                # (reps, T)
    reps, T = dev.shape
    H = np.zeros((T, n_bins), dtype=float)
    for t in range(T):
        vals = dev[:, t]
        ok = ~np.isnan(vals)
        if ok.any():
            cnts, _ = np.histogram(vals[ok], bins=n_bins, range=(0.0, 0.5))
            H[t] = cnts / ok.sum()
    return H

def sum_fit_data_from_wbar(wbar: np.ndarray, n_bins: int = 6, max_fitness = None):
    """
    Histogram of wbar in [1, max_fitness], across replicates per time point.
    Returns (H, max_fitness) with H of shape (T, n_bins).
    """
    wbar = _ensure_reps_by_time(wbar)      # (reps, T)
    reps, T = wbar.shape
    if max_fitness is None:
        with np.errstate(all='ignore'):
            max_fitness = float(np.nanmax(wbar))
        if not np.isfinite(max_fitness) or max_fitness <= 1.0:
            max_fitness = 1.01

    H = np.zeros((T, n_bins), dtype=float)
    for t in range(T):
        vals = wbar[:, t]
        ok = ~np.isnan(vals)
        if ok.any():
            cnts, _ = np.histogram(vals[ok], bins=n_bins, range=(1.0, max_fitness))
            H[t] = cnts / ok.sum()
    return H, max_fitness

def summarize_batch_for_npe(batch_tensor: torch.Tensor,
                            n_bins_freq: int = 5,
                            n_bins_fit: int = 6,
                            use_wbar_times = None) -> torch.Tensor:
    """
    batch_tensor: torch.tensor of shape (reps, 3*T) = [GFP | wbar | ubar].
    Returns a 1D torch tensor with concatenated histogram summaries.
    """
    X = batch_tensor.detach().cpu().numpy()
    reps, L = X.shape
    assert L % 3 == 0, "Expected concatenation [GFP | wbar | ubar]"
    T = L // 3
    GFP  = X[:, 0:T]
    wbar = X[:, T:2*T]
    # ubar = X[:, 2*T:3*T]  # not used in hist summaries (can be added if desired)

    H_freq = sum_freq_data_from_GFP(GFP, n_bins=n_bins_freq)       # (T, n_bins_freq)
    H_fit, max_fit = sum_fit_data_from_wbar(wbar, n_bins=n_bins_fit)  # (T, n_bins_fit)

    if use_wbar_times is not None:
        H_fit = H_fit[use_wbar_times, :]

    summary_vec = np.concatenate([H_freq, H_fit], axis=1)
    return torch.tensor(summary_vec, dtype=torch.float32)

# --- your simulator with summary output ---

def simulator(theta,
              N0=100000,
              gens_per_bottle=5,
              total_gens=300,
              reps=1,
              n_bins_freq: int = 5,
              n_bins_fit: int = 6,
              use_wbar_times = None,
              return_raw: bool = False):
    """
    Simulate `reps` trajectories, collect [GFP | wbar | ubar] at checkpoints,
    and return histogram-based summary statistics (and optionally raw outputs).

    Returns
    -------
    summary : torch.Tensor, shape (D,)
        Concatenated histogram features across time points.
    raw (optional) : torch.Tensor, shape (reps, 3*T)
        Raw concatenation per replicate: [GFP, wbar, ubar].
    """
    Ub, alpha, beta = theta
    Ub   = 10.0 ** Ub
    beta = 10.0 ** beta
    tau  = 1

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

        GFP  = calc_GFP(history)       # shape (T,)
        wbar = calc_wbar(history)      # shape (T,)
        ubar = calc_ubar(history)      # shape (T,)

        result = np.concatenate([GFP, wbar, ubar]).astype(np.float32)
        results.append(torch.tensor(result, dtype=torch.float32))

    batch = torch.stack(results, dim=0)             # (reps, 3*T)
    summary = summarize_batch_for_npe(batch,
                                      n_bins_freq=n_bins_freq,
                                      n_bins_fit=n_bins_fit,
                                      use_wbar_times=use_wbar_times)
    if return_raw:
        return summary, batch
    return summary

def simulator_UB(theta,
              N0=100000,
              gens_per_bottle=5,
              total_gens=300,
              reps=1,
              n_bins_freq: int = 5,
              n_bins_fit: int = 6,
              use_wbar_times = None,
              return_raw: bool = False):
    """Simultor with a fixed a=10"""

    theta_to_sim = np.array([theta[0],10,theta[1]])
    return simulator(theta_to_sim, N0, gens_per_bottle, total_gens, reps, n_bins_freq, n_bins_fit, use_wbar_times, return_raw)