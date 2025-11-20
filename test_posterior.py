# inference with NPE
from simulators import WF, GLU, SLCP, wrapper, wrapper_hierarchical
from evo_sim import evo_sim
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

from sbi.inference import MCMCPosterior

torch.set_num_threads(80)

# torch no grad
torch.set_grad_enabled(False)

#### arguments ####
parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model") # model name
parser.add_argument('-p', "--posterior") # posterior directory
parser.add_argument('-s', "--samples") # number of samples from the posterior
parser.add_argument('-t', "--test_thetas") # test thetas directory
parser.add_argument('-x', "--test_x") # test x directory
parser.add_argument('-cp', "--cp", action='store_true') # whether to use collective posterior
parser.add_argument('-e', "--ending", default='') # ending for file names
args = parser.parse_args()


model_dict = {'GLU': GLU, 'WF': WF, 'SLCP': SLCP, 'EVO_SIM': evo_sim}

# Define the prior and simulator
sim = str(args.model)
simulator = model_dict[sim]
thetas_dir = args.test_thetas
x_dir = args.test_x
posterior_dir = args.posterior
samples = int(args.samples)
c = args.cp
h = x_dir[-4] == 'h'
ending = args.ending

# Load the posterior with pickle
prior = get_prior(sim)
posterior = pickle.load(open(posterior_dir, 'rb'))

# if h and not c:
    # posterior = MCMCPosterior(posterior.potential_fn, proposal=prior)

conf_levels = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]

# Load the test thetas
thetas = torch.load(thetas_dir)
X = torch.load(x_dir)

def coverage_old(posterior, samples, conf_levels, theta):
    covs = torch.empty(len(conf_levels), len(theta))
    for j in range(len(conf_levels)):
        conf_level = conf_levels[j]   
        taken_samples = samples[:int(conf_level*len(samples))+1]
        hdi = [torch.quantile(samples, (1-conf_level)/2, 0), torch.quantile(samples,(1+conf_level)/2, 0)]
        covs[j,:] = ((theta > hdi[0])*(theta < hdi[1]))
    
    return covs

    
def evaluate(posterior, thetas, n_samples, cp = False):
    if sim == 'WF':
        epsilon = -150
    if sim == 'EVO_SIM':
        epsilon = -10
    else:
        epsilon = -10000
    accus = torch.empty(thetas.shape)
    covs = torch.empty(len(thetas[:,0]),len(conf_levels), len(thetas[0]))
    all_samples = torch.empty(len(thetas), n_samples, len(thetas[0]))
    for i in range(len(thetas)):
        th = thetas[i]
        x = X[i]
        if cp:
            cp = CollectivePosterior(prior, amortized_posterior=posterior, log_C=1, Xs=x, epsilon=epsilon)
            cp.get_log_C()
            samples = cp.sample(n_samples)
        else:
            samples = posterior.set_default_x(x).sample((n_samples,))
        all_samples[i,:,:] = samples
        params = torch.tensor(th, dtype=torch.float32)
        accus[i] = samples.mean(0)-params
        covs[i] = coverage_old(posterior, samples, conf_levels, theta=th)
        if i%10 == 9:
            print(f'{round(100*(i+1)/len(thetas),2)}%')
    return accus, covs, all_samples

add_iid = '' if c else '_iid'
add_h = '_h' if h else ''


accus, covs, all_samples = evaluate(posterior, thetas, n_samples=samples, cp=c)


torch.save(accus, f'{sim}/accus_{sim}{add_iid}{add_h}{ending}.pt')
torch.save(covs, f'{sim}/covs_{sim}{add_iid}{add_h}{ending}.pt')
torch.save(all_samples, f'{sim}/samples_{sim}{add_iid}{add_h}{ending}.pt')
