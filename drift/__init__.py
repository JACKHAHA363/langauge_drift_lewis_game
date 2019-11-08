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
EVALUATION_RATIO = 0.1 if not USE_GPU else 0.5
print('The currect evaluation ratio is {:.2f}. '.format(EVALUATION_RATIO))
