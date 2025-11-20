import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sbi.utils as utils
import torch
from scipy.special import logsumexp
from scipy.optimize import minimize
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Tuple, Dict


class CollectivePosterior:
    """
    A class to represent and sample from a collective posterior distribution.

    Parameters:
        prior: Prior distribution object compatible with `torch.distributions`.
        amortized_posterior: Pretrained posterior model (from sbi or other libraries).
        Xs: Set of observed data points.
        log_C: Normalizing constant for the posterior. Default is 1.
        epsilon: Sensitivity parameter, used as a lower bound for posterior probabilities. Default is -10000.
        n_eval: Number of samples used for Monte Carlo estimation. Default is 1e5.
        sample_var: Variance for MCMC-like sampling. Default is 0.05.
        from_sbi: Whether the posterior is from the sbi library. Default is True.
    """
       

    def __init__(self, prior, Xs, amortized_posterior=None, log_C=1, epsilon=-10000,
                 n_eval=int(1e5), sample_var=0.05, posterior_list=[]):
        self.prior = prior
        self.amortized_posterior = amortized_posterior
        self.Xs = Xs
        self.log_C = log_C
        self.epsilon = epsilon
        self.map = None
        self.samples = torch.empty((0, prior.sample().shape[0]))
        self.theta_dim = prior.sample().shape[0]
        self.n_eval = n_eval
        self.sample_var = sample_var
        self.posterior_list = posterior_list
        assert (len(posterior_list) > 0 or amortized_posterior is not None)


    def log_prob(self, theta):
        """
        Compute log q_coll(theta) = sum_i log q_i(theta|x_i) - (r-1) log p(theta) - log C
        Assumes self.log_C was set via get_log_C() or provided at init.
        """
        if self.log_C is None:
            raise RuntimeError("log_C is not set. Call get_log_C() first or pass it at init.")

        # to tensor, allow (d,) or (N,d)
        theta = torch.as_tensor(theta, dtype=torch.float32)
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)  # (1, d)

        N = theta.shape[0]
        r = len(self.Xs)

        # collect per-replicate log q_i
        log_probs = torch.empty((N, r), dtype=torch.float32)
        for i in range(r):
            if len(self.posterior_list) == 0:  # amortized posterior
                lp_i = self.amortized_posterior.set_default_x(self.Xs[i, :]).log_prob(theta)
            else:  # provided per-replicate log-posteriors
                lp_i = self.posterior_list[i](theta)  # expects (N,) or (N,1)
            log_probs[:, i] = torch.clamp(lp_i.reshape(N,), min=self.epsilon)  # epsilon is a LOG floor

        sum_logq = log_probs.sum(dim=1)                           # (N,)
        logp = self.prior.log_prob(theta).reshape(N,)             # (N,)

        return sum_logq - (r - 1) * logp - self.log_C


    def sample(self, n_samples, keep=True, step_size=0.05, take_sn=50, jump=1e5, m=1.5, num_workers=24, method='mcmc'):
        if method == 'mcmc':
            return self.mcmc_from_top_sn(n_total=n_samples, step_size=step_size, take_sn=take_sn)
        elif method == 'rejection':
            return self.rejection_sample(n_samples, jump=jump, m=m, keep=keep, n_workers=num_workers)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def mcmc_from_top_sn(self, n_total=1000, step_size=0.05, take_sn=50):
        """
        Run independent Metropolis-Hastings chains from each top SN sample in self.samples.
        n_total: total number of samples to generate (will be split across chains).
        Adds a progress bar for each chain and a global progress bar.
        Returns: tensor of shape (n_total, theta_dim)
        """
        top_sn_samples = self.samples[:take_sn]
        log_prob_fn = lambda theta: self.log_prob(theta)
        all_samples = []
        theta_dim = top_sn_samples.shape[1]
        n_per_chain = math.ceil(n_total / top_sn_samples.shape[0])
        total_steps = n_per_chain * top_sn_samples.shape[0]
        with tqdm(total=total_steps, desc=f"MCMC from top {take_sn} candidates", initial=take_sn) as global_pbar:
            for idx, start in enumerate(top_sn_samples):
                chain = [start.clone()]
                cur_logp = log_prob_fn(start.unsqueeze(0))[0]
                chain_pbar = tqdm(total=n_per_chain, desc=f"Chain {idx+1}/{take_sn}", leave=False)
                for _ in range(n_per_chain - 1):
                    proposal = chain[-1] + torch.randn(theta_dim) * step_size
                    prop_logp = log_prob_fn(proposal.unsqueeze(0))[0]
                    accept = torch.rand(1).item() < torch.exp(prop_logp - cur_logp)
                    if accept:
                        chain.append(proposal)
                        cur_logp = prop_logp
                    else:
                        chain.append(chain[-1].clone())
                    chain_pbar.update(1)
                    global_pbar.update(1)
                chain_pbar.close()
                all_samples.append(torch.stack(chain))
        samples = torch.cat(all_samples, dim=0)[:n_total]
        return samples


# parallel version of rejection sampling
    def rejection_sample(self, n_samples, jump=int(1e5), m = 1.5, keep=True, n_workers=48):
        """
        Sample from the collective posterior using parallel rejection sampling.

        Parameters:
            n_samples (int): Number of samples to generate.
            jump (int): Number of prior samples to draw in each batch. Default is 1e5.
            m (float): Offset for the acceptance criterion. Default is 5.
            keep (bool): Whether to store the samples in the object. Default is True.
            n_workers (int): Number of parallel workers. Default is 48.

        Returns:
            torch.Tensor: Samples from the posterior.
        """
        samples = torch.empty((1, self.theta_dim))
        cur = 0
        with tqdm(total=n_samples, desc="Rejection Sampling") as pbar:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                while cur < n_samples:
                    futures.append(executor.submit(self._rejection_sample_batch, jump, m))
                    # Collect results as they complete
                    for future in as_completed(futures):
                        samps, count = future.result()
                        samples = torch.cat([samples, samps])
                        cur += count
                        pbar.update(count)
                        futures.remove(future)
                        if cur >= n_samples:
                            break
        if keep:
            self.samples = samples[1:n_samples+1]
        return samples[1:n_samples+1]
    
    def _rejection_sample_batch(self, jump, m):
        samps = self.prior.sample((jump,))
        prior_probs = self.prior.log_prob(samps)
        probs = torch.log(torch.rand(samps.size()[0]))
        lp = self.log_prob(samps)
        next_idx = (lp - prior_probs) > (probs + m)
        samples_to_add = samps[next_idx]
        return samples_to_add, next_idx.sum().item()
    
    # def rejection_sample(self, n_samples, jump=int(1e4), m = 5, keep=True):
    #     """
    #     Sample from the collective posterior using rejection sampling.

    #     Parameters:
    #         n_samples (int): Number of samples to generate.
    #         jump (int): Number of prior samples to draw in each batch. Default is 1e5.
    #         keep (bool): Whether to store the samples in the object. Default is True.
        
    #     Returns:
    #         torch.Tensor: Samples from the posterior.
    #     """
    #     samples = torch.empty((1, self.theta_dim))
    #     cur = 0

    #     with tqdm(total=n_samples, desc="Rejection Sampling") as pbar:
    #         while cur < n_samples:
    #             samps = self.prior.sample((jump,))
    #             prior_probs = self.prior.log_prob(samps)
    #             probs = torch.log(torch.rand(samps.size()[0]))
    #             lp = self.log_prob(samps)
    #             next_idx = (lp - prior_probs) > (probs+m)
    #             samples_to_add = samps[next_idx]
    #             samples = torch.cat([samples, samples_to_add])
    #             cur += next_idx.sum()
    #             pbar.update(next_idx.sum().item())  # Update the progress bar

    #     if keep:
    #         self.samples = samples[1:n_samples+1]

    #     return samples[1:n_samples+1]
    
    def sample_one(self, jump=int(1e5), keep=False):
        """
        Draw a single sample from the posterior.

        Parameters:
            jump (int): Number of prior samples to draw in each batch. Default is 1e4.
            keep (bool): Whether to store the sample in the object. Default is True.
        
        Returns:
            torch.Tensor: A single sample from the posterior.
        """
        sampled=False
        while not(sampled):
            samps = self.prior.sample((jump,))
            probs = torch.rand(samps.size()[0])
            lp = self.log_prob(samps)
            next_idx = lp > probs
            if next_idx.sum()>0:
                sampled=True
        return samps[next_idx][0]
    
    def sample_around(self, theta, jump=int(1e4)):
        """
        Sample around a given point using a Gaussian proposal.

        Parameters:
            theta (torch.Tensor): Center of the Gaussian proposal.
            jump (int): Number of candidates to sample. Default is 1e4.
        
        Returns:
            tuple: (Accepted samples, new center theta).
        """
        dist =  torch.distributions.multivariate_normal.MultivariateNormal(theta, torch.diag(torch.tensor([self.sample_var]*self.theta_dim)))
        cands = dist.sample((jump,))
        probs = self.log_prob(cands)
        baseline = torch.rand((len(cands),))
        res = cands[probs>baseline]
        new_theta = cands[probs.argmax()]
        return res, new_theta
    
#     def sample_unimodal(self, n_samples, jump=int(1e5), keep=True):
#         """
#         Sample from the posterior using iterative proposals.

#         Parameters:
#             n_samples (int): Number of samples to generate.
#             jump (int): Number of candidates to sample at each iteration. Default is 1e4.
#             keep (bool): Whether to store the samples in the object. Default is True.
        
#         Returns:
#             torch.Tensor: Samples from the posterior.
#         """
#         theta = self.sample_one(jump, keep=False)
#         samples = torch.empty((n_samples, len(theta)))
#         cur = 0

#         with tqdm(total=n_samples, desc="Sampling") as pbar:
#             while cur < n_samples:
#                 samps, theta = self.sample_around(theta, jump)
#                 how_many = len(samps)
#                 if cur + how_many > n_samples:
#                     how_many = n_samples - cur
#                 if how_many > 0:
#                     samples[cur:cur + how_many, :] = samps[:how_many, :]
#                     cur += how_many
#                     pbar.update(how_many)  # Update the progress bar

#         if keep:
#             self.samples = samples

#         return samples

    def sample_multimodal(
        self,
        n_samples: int,                     # number of starting centres
        jump: int = 100_000,
        keep: bool = True,
        k: int = 10,
        T: float = 10.0                     # exploration threshold parameter
    ):
        """
        Draw `n_samples` points by running `k` independent short explorations.
        Threshold of exploration is log_prob of current centre - T.

        If self.samples is not empty, use those as centres.
        Each sub-run starts from a different seed drawn with `sample_one`.

        NOTE: correctness hinges on `sample_one` / `sample_around`
        being proper rejection samplers (see §2 below).
        """
        assert k >= 1, "k must be at least 1"
        per_chain = math.ceil(n_samples / k)      # sub-samples per centre
        all_chunks = []
        if len(self.samples) > 0:
            init_thetas = self.samples[torch.randint(0, len(self.samples), (k,))]
        else:
            init_thetas = self.rejection_sample(k, jump, keep)
        with tqdm(total=n_samples, desc="Sampling") as bar:
            for j in range(k):
                # 1) independent seed
                theta = init_thetas[j]
                chunk = torch.empty((per_chain, self.theta_dim))
                cur = 0
                # Set threshold for exploration
                threshold = self.log_prob(theta) - T
                while cur < per_chain:
                    samps, new_theta = self.sample_around(theta, jump)
                    # Only keep samples above the threshold
                    mask = self.log_prob(samps) > threshold
                    filtered_samps = samps[mask]
                    take = min(per_chain - cur, filtered_samps.size(0))
                    if take:
                        chunk[cur : cur + take] = filtered_samps[:take]
                        cur += take
                        bar.update(take)
                    theta = new_theta
                all_chunks.append(chunk)
        samples = torch.cat(all_chunks, dim=0)[:n_samples]   # exact count
        if keep:
            self.samples = samples
        return samples

    
    def get_map(self, n_init=100, keep=True):
        """
        Compute the Maximum A Posteriori (MAP) estimate.

        Parameters:
            keep (bool): Whether to store the MAP estimate in the object. Default is True.
        
        Returns:
            torch.Tensor: MAP estimate of the posterior mode.
        """
        samples = self.samples
        
        if len(self.posterior_list) > 0:
            func = lambda theta: np.array(-1*self.log_prob(theta)).reshape(1,-1) 
        else:
            func = lambda theta: np.array(-1*self.log_prob(theta)) # max(log_prob) = min(-log_prob) + np array for optimizer
        if len(self.samples) > 0:
            x0 = self.samples[self.log_prob(samples).argmax()]
        else:
            samples = self.sample(n_init)
            x0 = self.samples[self.log_prob(samples).argmax()] # get a nice random guess
        x0 = list(x0)
        collective_map = torch.from_numpy(minimize(func, x0, method='Nelder-Mead').x) # Optimization
        if keep:
            self.map = collective_map
        return collective_map
        
    
    def get_log_C(self, n_reps=10, S=0.005):
        """
        Estimate log C = log ∫ u(θ) dθ via importance sampling with θ ~ p(θ),
        where u(θ) = Π_i q_i(θ|x_i) / p(θ)^{r-1}.
        Monte Carlo identity:
            C = E_{θ~p} [ exp( Σ_i log q_i(θ|x_i) - r * log p(θ) ) ].
        We average in probability space across repetitions, then take log.
        """
        r = len(self.Xs)
        N = int(self.n_eval)

        rep_logs = []
        for _ in range(n_reps):
            theta = self.prior.sample((N,))                       # (N,d)
            prior_logps = self.prior.log_prob(theta).reshape(N,)  # (N,)

            # build (N, r) matrix of log q_i
            log_probs = torch.empty((N, r), dtype=torch.float32)
            for i in range(r):
                if len(self.posterior_list) == 0:
                    lp_i = self.amortized_posterior.set_default_x(self.Xs[i, :]).log_prob(theta)
                else:
                    lp_i = self.posterior_list[i](theta)
                log_probs[:, i] = torch.clamp(lp_i.reshape(N,), min=self.epsilon)  # LOG floor

            sum_logq = log_probs.sum(dim=1)                       # (N,)
            # importance weights in log-space: log u(θ) - log p(θ) = Σ log q_i - r log p
            weights_log = sum_logq - r * prior_logps              # (N,)
            # log-mean-exp over N samples
            logC_rep = torch.logsumexp(weights_log, dim=0) - math.log(N)
            rep_logs.append(logC_rep)
            # add top S samples in self.samples
            topk = weights_log.topk(int(S*N)).indices

            self.samples = torch.cat([self.samples, theta[topk]], dim=0)

        # average across repetitions in probability space, then log
        self.log_C = torch.logsumexp(torch.stack(rep_logs), dim=0) - math.log(len(rep_logs))
        # sort samples by log prob
        if len(self.samples) > 0:
            sorted_indices = self.log_prob(self.samples).argsort(descending=True)
            self.samples = self.samples[sorted_indices]

        return self.log_C

