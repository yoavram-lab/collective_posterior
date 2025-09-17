# inference with NPE
from simulators import WF_wrapper, GLU_wrapper, SLCP_wrapper, GORDO_wrapper
from inference_utils import get_prior


from sbi.utils import BoxUniform
import torch
from joblib import Parallel, delayed
import pickle
from time import time
import argparse
import sbibm
from sbi.inference import NPE, simulate_for_sbi

# Suppress warnings containing 'NaN'
import warnings
warnings.filterwarnings("ignore", message=".*NaN.*")
warnings.filterwarnings("ignore", message=".*torch.*")


# Ensure PyTorch uses 100 CPU cores
torch.set_num_threads(100)

from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)



# time
start = time()

# Define the prior and simulator
#### arguments ####
parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model")
parser.add_argument('-e', "--epochs")
parser.add_argument('-n', "--num_sim")
args = parser.parse_args()




stop_after_epochs = int(args.epochs)
num_sim = int(args.num_sim)

sim = str(args.model)
prior = get_prior(sim)
model_dict = {'GLU': GLU_wrapper, 'WF': WF_wrapper, 'SLCP': SLCP_wrapper, 'GORDO': GORDO_wrapper}
simulator = model_dict[sim]


max_num_trials = 10
  

# construct training data set: we want to cover the full range of possible number of
# trials
num_training_samples = num_sim
theta = prior.sample((num_training_samples,))

# there are certainly smarter ways to construct the training data set, but we go with a
# for loop here for illustration purposes.
x_dim_dict = {'GLU': 10, 'WF': 12, 'SLCP': 8, 'GORDO': 39} 
x_dim = x_dim_dict[sim]


# Parallelized simulation using joblib
def simulate_and_fill(i):
    xi = simulator(reps=max_num_trials, parameters=theta[i])
    rows = []
    for j in range(max_num_trials):
        row = torch.full((max_num_trials, x_dim), float('nan'))
        row[:j+1, :] = xi[:j+1, :]
        rows.append(row)
    return rows

results = Parallel(n_jobs=100)(delayed(simulate_and_fill)(i) for i in range(num_training_samples))
x = torch.stack([row for sublist in results for row in sublist])

theta = theta.repeat_interleave(max_num_trials, dim=0)
torch.save(theta, f'{sim}/theta_train_iid.pt')
print(f'Saved theta to {sim}/theta_train_iid.pt')
# inference
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding

# embedding
latent_dim = 10
single_trial_net = FCEmbedding(
    input_dim=x_dim,
    num_hiddens=40,
    num_layers=2,
    output_dim=latent_dim,
)
embedding_net = PermutationInvariantEmbedding(
    single_trial_net,
    trial_net_output_dim=latent_dim,
    # NOTE: post-embedding is not needed really.
    num_layers=1,
    num_hiddens=10,
    output_dim=10,
)

# we choose a simple MDN as the density estimator.
# NOTE: we turn off z-scoring of the data, as we used NaNs for the missing trials.
density_estimator = posterior_nn("maf", embedding_net=embedding_net, z_score_x="none")

inference = NPE(prior, density_estimator=density_estimator)
# NOTE: we don't exclude invalid x because we used NaNs for the missing trials.
inference.append_simulations(
    theta,
    x,
    exclude_invalid_x=False,
).train(stop_after_epochs=stop_after_epochs)
posterior = inference.build_posterior()

# Save the posterior with pickle
with open(f'{sim}/posterior_iid_{num_sim}.pkl', 'wb') as f:
    pickle.dump(posterior, f)

# time
end = time()
print(f' Inference took {end - start} seconds')
