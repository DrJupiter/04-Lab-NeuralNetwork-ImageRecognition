import torch
import numpy as np

"""
y = [fish,not_fish]
y_prime = [% fish, % not_fish]
"""
Loss_fn = torch.nn.MSELoss(reduction='sum')

