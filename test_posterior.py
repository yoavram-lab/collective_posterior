# inference with NPE
from simulators import WF, GLU, SLCP, wrapper, wrapper_hierarchical, CLASSIC_WF
import torch
import pickle
# import time
import argparse
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import numpy as np
import pandas as pd
from collective_posterior import CollectivePosterior
from inference_utils import get_prior

from sbi.inference import ImportanceSamplingPosterior, MCMCPosterior

torch.set_num_threads(80)

# torch no grad
torch.set_grad_enabled(False)

#### arguments ####
parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model") # model name
parser.add_argument('-p', "--posterior") # posterior directory
parser.add_argument('-s', "--samples") # number of samples from the posterior
parser.add_argument('-t', "--test_thetas") # test thetas directory
parser.add_argument('-hi', "--hierarchical", action='store_true') # whether to test on hierarchical model
parser.add_argument('-cp', "--cp", action='store_true') # whether to use collective posterior
parser.add_argument('-e', "--ensemble", action='store_true') # whether to test on an ensemble
parser.add_argument('-ss', "--save_samples", action='store_true') # whether to save samples
args = parser.parse_args()


model_dict = {'GLU': GLU, 'WF': WF, 'SLCP': SLCP, 'CLASSIC_WF': CLASSIC_WF}

# Define the prior and simulator
sim = str(args.model)
simulator = model_dict[sim]
thetas_dir = args.test_thetas
posterior_dir = args.posterior
samples = int(args.samples)
c = args.cp
h = args.hierarchical
e = args.ensemble
ss = args.save_samples

# Load the posterior with pickle
prior = get_prior(sim)
posterior = pickle.load(open(posterior_dir, 'rb'))
# posterior = MCMCPosterior(posterior.potential_fn, prior)

conf_levels = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]

# Load the test thetas
thetas = torch.tensor(np.array(pd.read_csv(thetas_dir, index_col=0).values.astype('float')), dtype=torch.float32)
# thetas = torch.load(thetas_dir)


def coverage_old(posterior, samples, conf_levels, theta):
    covs = torch.empty(len(conf_levels), len(theta))
    for j in range(len(conf_levels)):
        conf_level = conf_levels[j]   
        taken_samples = samples[:int(conf_level*len(samples))+1]
        hdi = [torch.quantile(samples, (1-conf_level)/2, 0), torch.quantile(samples,(1+conf_level)/2, 0)]
        covs[j,:] = ((theta > hdi[0])*(theta < hdi[1]))
    
    return covs

    
def info_gain(posterior, samples, conf_levels):
    res = torch.empty(len(conf_levels))
    for i in range(len(conf_levels)):
        conf_level = conf_levels[i]
        ql, qh = (1-conf_level)/2, (1+conf_level)/2
        res[i] = (samples.quantile(qh, dim=0) - samples.quantile(ql, dim=0)).prod() / (prior.base_dist.high-prior.base_dist.low).prod()
    return res

def evaluate_cp(posterior, thetas, n_samples):
    n_set = 10
    if sim == 'WF':
        epsilon = -150
    if sim == 'CLASSIC_WF':
        epsilon = -10
    else:
        epsilon = -10000
    accus = torch.empty(thetas.shape)
    covs = torch.empty(len(thetas),1)
    covs_old = torch.empty(len(thetas[:,0]),len(conf_levels), len(thetas[0]))
    ig = torch.empty(len(thetas),len(conf_levels))
    log_probs = torch.empty(len(thetas),1)
    all_samples = torch.empty(len(thetas), n_samples, len(thetas[0]))
    for i in range(len(thetas)):
        th = thetas[i]
        if h:
            x_h = wrapper_hierarchical(simulator, reps=3, parameters=th, var=0.2)
            x = wrapper(simulator, reps=n_set-3, parameters=th)
            x = torch.cat((x_h, x), dim=0)
        else:
            x = wrapper(simulator, reps=n_set, parameters=th)
        cp = CollectivePosterior(prior, amortized_posterior=posterior, log_C=1, Xs=x, epsilon=epsilon)
        cp.get_log_C()
        samples = cp.mcmc_from_top_sn(n_samples, take_sn=50)
        if ss:
            all_samples[i,:,:] = samples
        params = torch.tensor(th, dtype=torch.float32)
        accus[i] = samples.mean(0)-params
        covs[i] = (cp.log_prob(samples) > cp.log_prob(params)).sum()/n_samples
        covs_old[i] = coverage_old(posterior, samples, conf_levels, theta=th)
        ig[i] = info_gain(posterior, samples, conf_levels)
        log_probs[i] = cp.log_prob(th).item()
        if i%10 == 9:
            print(f'{round(100*(i+1)/len(thetas),2)}%')
    return accus, covs, covs_old, ig, log_probs, all_samples

def evaluate_iid(posterior, thetas, n_samples):
    n_set = 10
    accus = torch.empty(thetas.shape)
    covs = torch.empty(len(thetas),1)
    covs_old = torch.empty(len(thetas[:,0]),len(conf_levels), len(thetas[0]))
    ig = torch.empty(len(thetas),len(conf_levels))
    log_probs = torch.empty(len(thetas),1)
    all_samples = torch.empty(len(thetas), len(thetas[0])*n_samples)
    for i in range(len(thetas)):
        th = thetas[i]
        if h:
            x = wrapper_hierarchical(simulator, reps=n_set, parameters=th)
        else:
            x_h = wrapper_hierarchical(simulator, reps=3, parameters=th, var=0.2)
            x = wrapper(simulator, reps=n_set-3, parameters=th)
            x = torch.cat((x_h, x), dim=0)
        
        samples = posterior.set_default_x(x).sample((n_samples,))
        if ss:
            all_samples[i,:] = samples.T.flatten()

        params = torch.tensor(th, dtype=torch.float32)
        accus[i] = samples.mean(0)-params
        covs[i] = (posterior.potential(samples) > posterior.potential(params)).sum()/len(thetas)
        covs_old[i] = coverage_old(posterior, samples, conf_levels, theta=th)
        ig[i] = info_gain(posterior, samples, conf_levels)
        log_probs[i] = posterior.log_prob(th).item()
        if i%10 == 9:
            print(f'{round(100*i/len(thetas),2)}%')
    return accus, covs, covs_old, ig, log_probs, all_samples

eval_func = evaluate_cp if c else evaluate_iid
add_iid = '' if c else '_iid'
add_h = '_h' if h else ''
add_e = '_e' if e else ''


accus, covs, covs_old, ig, log_probs, all_samples = eval_func(posterior, thetas, n_samples=samples)
covs_old = covs_old.mean(0)
accus = accus.detach().numpy()
covs = covs.detach().numpy()
covs_old = covs_old.detach().numpy()
ig = ig.detach().numpy()
log_probs = log_probs.detach().numpy()
all_samples = all_samples.detach().numpy()


pd.DataFrame(accus).to_csv(f'{sim}/tests/accus_{sim}{add_iid}{add_h}{add_e}.csv')
pd.DataFrame(covs).to_csv(f'{sim}/tests/covs_{sim}{add_iid}{add_h}{add_e}.csv')
pd.DataFrame(covs_old).to_csv(f'{sim}/tests/covs_old_{sim}{add_iid}{add_h}{add_e}.csv')
pd.DataFrame(ig).to_csv(f'{sim}/tests/ig_{sim}{add_iid}{add_h}{add_e}.csv')
pd.DataFrame(log_probs).to_csv(f'{sim}/tests/logprobs_{sim}{add_iid}{add_h}{add_e}.csv')
if ss:
    pd.DataFrame(all_samples).to_csv(f'{sim}/tests/samples_{sim}{add_iid}{add_h}{add_e}.csv')
