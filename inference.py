# inference with NPE
from simulators import WF, GLU, SLCP
from evo_sim import evo_sim
from inference_utils import get_prior
import torch
import pickle
import time
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils import BoxUniform
import argparse
import sbibm
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

torch.set_num_threads(80)

#### arguments ####
parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model")
parser.add_argument('-e', "--epochs", default=20)
parser.add_argument('-n', "--num_sim")
args = parser.parse_args()

# time
start = time.time()

model_dict = {'GLU': GLU, 'WF': WF, 'SLCP': SLCP, 'EVO_SIM': evo_sim}

# Define the prior and simulator
sim = str(args.model)
stop_after_epochs = int(args.epochs)
prior = get_prior(sim)
simulator = model_dict[sim]
num_sim = int(args.num_sim)

# inference
# Check prior, return PyTorch prior.
prior, num_parameters, prior_returns_numpy = process_prior(prior)
theta = prior.sample((num_sim,))
# Check simulator, returns PyTorch simulator able to simulate batches.
simulator = process_simulator(simulator, prior, prior_returns_numpy)

# inference
inference = NPE(prior)
theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_sim, num_workers=80)
density_estimator = inference.append_simulations(theta, x).train(stop_after_epochs=stop_after_epochs)
posterior = inference.build_posterior(density_estimator)

# Save the posterior with pickle
with open(f'{sim}/posterior_{sim}_{num_sim}_{stop_after_epochs}.pkl', 'wb') as f:
    pickle.dump(posterior, f)

# time
end = time.time()
print(f'Inference time: {end - start} seconds')
