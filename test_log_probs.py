# inference with NPE
from simulators import WF, GLU, SLCP, wrapper, wrapper_hierarchical
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


model_dict = {'GLU': GLU, 'WF': WF, 'SLCP': SLCP, 'GORDO': GORDO}

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
posterior = pickle.load(open(posterior_dir, 'rb'))

# Load the test thetas
thetas = torch.tensor(pd.read_csv(thetas_dir, index_col=0).values.astype('float'), dtype=torch.float32)

def evaluate_cp(posterior, thetas, n_samples):
    n_set = 10
    if sim == 'WF':
        epsilon = -10
    else:
        epsilon = -10000
    log_probs = torch.empty(len(thetas),1)
    for i in range(len(thetas)):
        if h:
            X = wrapper_hierarchical(simulator, n_set, thetas[i])
        else:
            X = wrapper(simulator, n_set, thetas[i])
        cp = CollectivePosterior(prior=get_prior(sim), amortized_posterior=posterior, log_C=1, Xs=X, epsilon=epsilon)
        cp.get_log_C()
        log_probs[i] = cp.log_prob(thetas[i]).item()
        if i%10 == 1:
            print(f'{round(100*i/len(thetas),2)}%')
            print(log_probs[:i].mean())
    return log_probs

def evaluate_iid(posterior, thetas, n_samples):
    n_set = 10
    log_probs = torch.empty(len(thetas),1)
    for i in range(len(thetas)):
        if h:
            X = wrapper_hierarchical(simulator, n_set, thetas[i])
        else:
            X = wrapper(simulator, n_set, thetas[i])
        log_probs[i] = posterior.set_default_x(X).log_prob(thetas[i]).item()
        if i%10 == 1:
            print(f'{round(100*i/len(thetas),2)}%')
    print(log_probs.mean())
    return log_probs


eval_func = evaluate_cp if c else evaluate_iid
add_iid = '' if c else '_iid'
add_h = '_h' if h else ''
add_e = '_e' if e else ''
lp = eval_func(posterior, thetas, n_samples=1)
pd.DataFrame(lp).to_csv(f'{sim}/tests/logprobs_{sim}{add_iid}{add_h}{add_e}_r.csv')
