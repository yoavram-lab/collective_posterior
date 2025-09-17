from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pickle
from sbi.inference import NPE
from sbi.utils import BoxUniform
from wf import WF, combined_WF

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n', "--num_sim")
parser.add_argument('-ns', '--noisy', action='store_true')
parser.add_argument('-p', '--n_params')
args = parser.parse_args()

n_sim = int(args.num_sim)
noise = args.noisy
n_params = int(args.n_params)

# extract generations from the data
data = pd.read_csv('clean_data_exp_2.csv', index_col=0)
generation = np.array(data[data['Strain'] == 'DGY1886_1']['Generation'])

if n_params == 8:
    prior = BoxUniform(low=torch.tensor([-3, -5, -5, -3, -5, -5, -3, -5]), high=torch.tensor([-1, -2, -0.5, -1, -2, -0.5, -1, -0.5])) #logS, logDelta
    theta = prior.sample((n_sim,)).numpy()
    x = np.zeros((len(theta), 2*len(generation)))
    for i in range(len(theta)):
        s_cherry, m_cherry, p0_cherry, s_citrine, m_citrine, p0_citrine, s_cc, p0_cc = theta[i]
        x[i] = combined_WF(s_cherry, m_cherry, p0_cherry, s_citrine, m_citrine, p0_citrine, s_cc, p0_cc, generation, noisy=noise)

elif n_params==3:
    prior = BoxUniform(low=torch.tensor([-3, -5, -5]), high=torch.tensor([-1, -2, -0.5])) #logS, logDelta
    theta = prior.sample((n_sim,)).numpy()
    x = np.zeros((len(theta), len(generation)))
    for i in range(len(theta)):
        s_cnv, m_cnv, p0 = theta[i]
        x[i] = WF(s_cnv, m_cnv, p0, generation, noisy=noise)


# inference
inference = NPE(prior)
density_estimator = inference.append_simulations(torch.tensor(theta, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)).train()
posterior = inference.build_posterior(density_estimator)

# Save the posterior with pickle
add_noise = ''
if noise:
    add_noise = '_noisy'
with open(f'posterior_de.pkl', 'wb') as f:
    pickle.dump(posterior, f)
