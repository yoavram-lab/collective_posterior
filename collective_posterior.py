# Collective Posterior Distribution - PyTorch implementation
import torch
import math

# Optimizers for Maximum A-Posteriori (MAP)
import numpy as np
from scipy.optimize import minimize

# Progress bars
from tqdm import tqdm

# Parallelization (for rejection sampling)
from concurrent.futures import ThreadPoolExecutor, as_completed


class CollectivePosterior:
    """
    A class to represent and sample from a collective posterior distribution.

    Parameters:
        prior: Prior distribution object compatible with `torch.distributions`.
        amortized_posterior: Pretrained posterior model (from sbi or other libraries).
        Xs: Set of observed data points.
        log_C: Normalizing constant for the posterior. Default is 1. Should be estimated using get_log_C()
        epsilon: Sensitivity parameter, used as a lower bound for posterior probabilities. Default is -10000. Can be estimated as described in paper. 
        n_eval: Number of samples used for Monte Carlo estimation. Default is 1e5.
        sample_var: Variance for MCMC-like sampling. Default is 0.05.
        posterior_list: In case of using non-amortized posteriors, a list of log_prob functions can be used instead.
    """
       

    def __init__(self, prior, Xs, amortized_posterior=None, log_C=None, epsilon=-10000,
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
        
        # Ensure inference is possible
        assert (len(posterior_list) > 0 or amortized_posterior is not None)


    def log_prob(self, theta):
        """
        Compute log p_collective(theta) = sum_i log p_i(theta|x_i) - (r-1) log p(theta) - log C
        Assumes self.log_C was set via get_log_C() or provided at init.
        """
        if self.log_C is None:
            raise RuntimeError("log_C is not set. Call get_log_C() first or pass it at init.")

        # to tensor, allow (d,) or (N,d)
        theta = torch.as_tensor(theta, dtype=torch.float32)
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)  # (1, d)

        N = theta.shape[0] # parameters
        r = len(self.Xs) #replicates

        # collect per-replicate log p_i
        log_probs = torch.empty((N, r), dtype=torch.float32)
        for i in range(r):
            if len(self.posterior_list) == 0:  # amortized posterior
                lp_i = self.amortized_posterior.set_default_x(self.Xs[i, :]).log_prob(theta)
            else:  # provided per-replicate log-probs
                lp_i = self.posterior_list[i](theta)  # expects (N,) or (N,1)
            log_probs[:, i] = torch.clamp(lp_i.reshape(N,), min=self.epsilon)  # epsilon is a LOG floor

        sum_logp = log_probs.sum(dim=1)                           # (N,)
        logp = self.prior.log_prob(theta).reshape(N,)             # (N,)

        return sum_logp - (r - 1) * logp - self.log_C # log p_collective(theta) = sum_i log p_i(theta|x_i) - (r-1) log p(theta) - log C


    def sample(self, n_samples, keep=True, step_size=0.05, take_sn=50, jump=1e5, m=1.5, num_workers=24, method='mcmc'):
        if method == 'mcmc':
            return self.mcmc_from_top_sn(n_total=n_samples, step_size=step_size, take_sn=take_sn)
        elif method == 'rejection':
            return self.rejection_sample(n_samples, jump=jump, m=m, keep=keep, n_workers=num_workers)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def mcmc_from_top_sn(self, n_total=1000, step_size=0.05, take_sn=50):
        """
        Run independent Metropolis-Hastings chains from each top SN samples (by log-probs) in self.samples.
        n_total: total number of samples to generate (will be split across chains).
        Includes a global progress bar.
        Returns: tensor of shape (n_total, theta_dim)
        """
        assert len(self.samples) >= take_sn # enough samples to start

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
                # chain_pbar = tqdm(total=n_per_chain, desc=f"Chain {idx+1}/{take_sn}", leave=False)
                for _ in range(n_per_chain - 1):
                    proposal = chain[-1] + torch.randn(theta_dim) * step_size
                    prop_logp = log_prob_fn(proposal.unsqueeze(0))[0]
                    accept = torch.rand(1).item() < torch.exp(prop_logp - cur_logp)
                    if accept:
                        chain.append(proposal)
                        cur_logp = prop_logp
                    else:
                        chain.append(chain[-1].clone())
                    # chain_pbar.update(1)
                    global_pbar.update(1)
                # chain_pbar.close()
                all_samples.append(torch.stack(chain))
        samples = torch.cat(all_samples, dim=0)[:n_total]
        return samples


# parallel version of rejection sampling
    def rejection_sample(self, n_samples, jump=int(1e5), m = 1.5, keep=True, n_workers=8):
        """
        Sample from the collective posterior using parallel rejection sampling.

        Parameters:
            n_samples (int): Number of samples to generate.
            jump (int): Number of prior samples to draw in each batch. Default is 1e5.
            m (float): Offset for the acceptance criterion. Default is 5.
            keep (bool): Whether to store the samples in the object. Default is True.
            n_workers (int): Number of parallel workers. Default is 8.

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
    
    # rejection sampling helper
    def _rejection_sample_batch(self, jump, m):
        samps = self.prior.sample((jump,))
        prior_probs = self.prior.log_prob(samps)
        probs = torch.log(torch.rand(samps.size()[0]))
        lp = self.log_prob(samps)
        next_idx = (lp - prior_probs) > (probs + m)
        samples_to_add = samps[next_idx]
        return samples_to_add, next_idx.sum().item()
    
    


    
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
        where u(θ) = Π_i p_i(θ|x_i) / p(θ)^{r-1}.
        Monte Carlo identity:
            C = E_{θ~p} [ exp( Σ_i log p_i(θ|x_i) - r * log p(θ) ) ].
        We average in probability space across n_reps repetitions, then take log.
        We also keep S * self.n_eval samples as initializers for MCMC sampling. Defaults yield 5000.
        """
        r = len(self.Xs)
        N = int(self.n_eval)

        rep_logs = []
        for _ in range(n_reps):
            theta = self.prior.sample((N,))                       # (N,d)
            prior_logps = self.prior.log_prob(theta).reshape(N,)  # (N,)

            # build (N, r) matrix of log p_i
            log_probs = torch.empty((N, r), dtype=torch.float32)
            for i in range(r):
                if len(self.posterior_list) == 0:
                    lp_i = self.amortized_posterior.set_default_x(self.Xs[i, :]).log_prob(theta)
                else:
                    lp_i = self.posterior_list[i](theta)
                log_probs[:, i] = torch.clamp(lp_i.reshape(N,), min=self.epsilon)  # LOG floor

            sum_logp = log_probs.sum(dim=1)                       # (N,)
            # importance weights in log-space: log u(θ) - log p(θ) = Σ log p_i - r log p
            weights_log = sum_logp - r * prior_logps              # (N,)
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

