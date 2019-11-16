"""
Some global settings
"""
import torch

USE_GPU = torch.cuda.device_count() > 0
if USE_GPU:
    print('Use GPU')
else:
    print('Use CPU')

# Ratio of total objects used for evaluation [0,1]. Use all if we have GPU
EVALUATION_RATIO = 1 if not USE_GPU else 0.5
print('The currect evaluation ratio is {:.2f}. '.format(EVALUATION_RATIO))

# Generate the distribution for sampling
loc = torch.tensor(0.)
scale = torch.tensor(1.)
if USE_GPU:
    GUMBEL_DIST = torch.distributions.Gumbel(loc=loc.cuda(), scale=scale.cuda())
else:
    GUMBEL_DIST = torch.distributions.Gumbel(loc=loc, scale=scale)

