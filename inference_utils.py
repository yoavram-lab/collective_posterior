import torch
from sbi.utils import BoxUniform
import sbibm

# Define the prior
# Define the prior
def get_prior(sim):
    if sim == 'WF':
        prior = BoxUniform(low=torch.tensor([-2, -7, -8]), high=torch.tensor([0, -2, -2]))
    elif sim == 'GLU':
        prior = sbibm.get_task('gaussian_linear_uniform').get_prior_dist()
    elif sim == 'SLCP':
        prior = sbibm.get_task('slcp').get_prior_dist()
    elif sim == 'EVO_SIM':
        prior_low = torch.tensor([-3.0, -3.0, -3.0, -8.0, -8.0, -8.0])
        prior_high = torch.tensor([-1, -1, -1, -4, -4, -4])
        prior = BoxUniform(low=prior_low, high=prior_high)
    else:
        raise ValueError('Unknown simulator')

    return prior