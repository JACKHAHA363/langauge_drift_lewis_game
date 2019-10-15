"""
Some global settings
"""
import torch

USE_GPU = torch.cuda.is_available()

# Ratio of total objects used for evaluation [0,1]. Use all if we have GPU
EVALUATION_RATIO = 0.005 if not USE_GPU else 1.