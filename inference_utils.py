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
    elif sim == 'FWDPY':
        prior = BoxUniform(low=torch.tensor([-2, -6, 1]), high=torch.tensor([-0.5, -3, 4]))
    elif sim == 'CLASSIC_WF':
        prior = BoxUniform(low=torch.tensor([-3.0, -9, 5]), high=torch.tensor([-0.5, -4, 8]))
    else:
        raise ValueError('Unknown simulator')

    return prior