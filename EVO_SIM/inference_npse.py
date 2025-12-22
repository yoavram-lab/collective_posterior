import pickle
import time
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
import argparse

import sbi
print(sbi.__version__)
from sbi.inference import NPSE, simulate_for_sbi
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

sys.path.insert(1, '../')
from collective_posterior import CollectivePosterior
from evo_sim import evo_sim

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-t', "--test_thetas") # test thetas directory
parser.add_argument('-x', "--test_x") # test x directory
args = parser.parse_args()

thetas_dir = args.test_thetas
x_dir = args.test_x
samples = 10_000
h = x_dir[-4] == 'h'
n = x_dir[-4] == 'n'
r = x_dir[-4] == 'r'
e = x_dir[-4] == 'e'

# Check prior, return PyTorch prior.
prior_low = torch.tensor([-3.0, -3.0, -3.0, -8.0, -8.0, -8.0])
prior_high = torch.tensor([-1, -1, -1, -4, -4, -4])
prior = BoxUniform(low=prior_low, high=prior_high)
prior, num_parameters, prior_returns_numpy = process_prior(prior)

simulator = evo_sim

# Check simulator, returns PyTorch simulator able to simulate batches.
simulator = process_simulator(simulator, prior, prior_returns_numpy)

# Consistency check after making ready for sbi.
check_sbi_inputs(simulator, prior)

num_sims = 30_000

inference = NPSE(prior, sde_type='vp')

theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_sims, num_workers=80)
density_estimator = inference.append_simulations(theta, x).train()
posterior_npse = inference.build_posterior()

train_time = time.time()
elapsed = train_time - start_time
print(f"\n{'='*60}")
print(f"Total train time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
print(f"{'='*60}")

test_x_1000 = torch.load('test_x_1000.pt')
posterior_npse.set_default_x(test_x_1000).sample((samples,))





# thetas = torch.load(thetas_dir)
# X = torch.load(x_dir)

# h = x_dir[-4] == 'h'
# n = x_dir[-4] == 'n'
# r = x_dir[-4] == 'r'
# e = x_dir[-4] == 'e'


# conf_levels = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95] # for coverage
# def coverage_old(samples, conf_levels, theta):
#     covs = torch.empty(len(conf_levels), len(theta))
#     for j in range(len(conf_levels)):
#         conf_level = conf_levels[j]   
#         hdi = [torch.quantile(samples, (1-conf_level)/2, 0), torch.quantile(samples,(1+conf_level)/2, 0)]
#         covs[j,:] = ((theta > hdi[0])*(theta < hdi[1]))
    
#     return covs



# def evaluate(posterior, thetas, n_samples):
#     accus = torch.empty(thetas.shape)
#     covs = torch.empty(len(thetas[:,0]),len(conf_levels), len(thetas[0]))
#     all_samples = torch.empty(len(thetas), n_samples, len(thetas[0]))
#     for i in range(len(thetas)):
#         th = thetas[i]
#         x = X[i]
#         samples = posterior.set_default_x(x).sample((n_samples,), iid_method="gauss")
#         all_samples[i,:,:] = samples
#         params = torch.tensor(th, dtype=torch.float32)
#         accus[i] = samples.mean(0)-params
#         covs[i] = coverage_old(samples, conf_levels, theta=th)
#         if i%10 == 9:
#             print(f'{round(100*(i+1)/len(thetas),2)}%')
#     return accus, covs, all_samples




# accus, covs, all_samples = evaluate(posterior_npse, thetas, n_samples=samples)


# # Saving details

# add_h = '_h' if h else ''
# add_n = '_n' if n else ''
# add_r = '_r' if r else ''
# add_e = '_e' if e else ''
# sim = '.'
# ending = ''
# add_iid = '_npse'

# torch.save(accus, f'accus_{add_iid}{add_h}{add_n}{add_r}{add_e}{ending}.pt')
# torch.save(covs, f'covs_{add_iid}{add_h}{add_n}{add_r}{add_e}{ending}.pt')
# torch.save(all_samples, f'samples_{add_iid}{add_h}{add_n}{add_r}{add_e}{ending}.pt')


# test_time = time.time()
# elapsed = test_time - train_time
# print(f"\n{'='*60}")
# print(f"Total test time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
# print(f"{'='*60}")


