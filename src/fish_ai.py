import torch
import numpy as np

"""
y = [fish,not_fish]
y_prime = [% fish, % not_fish]
"""
Loss_fn = torch.nn.MSELoss(reduction='sum')

print("")

H = 10

# 10 -> 2
# 256,256, 2

# img
# torch.flatten(img) -> 256*256*2 = 131072
# torch.nn.Linear(131072, H)

#TODO størrelsen af imput vektoren afhænger af fildata, så hvis vi gerne vil bevare størrelsesforholdet  billedet, bliver vi nødt til at lade imputtet
#til vores NN være afhængingt at billdet, medmindre alt vores data har samme dimensioner. Det har jeg ingen anelse om

model = torch.nn.Sequential(
torch.nn.Linear(2,H),
torch.nn.Tanh(),
torch.nn.Linear(H,H),
torch.nn.Tanh(),
torch.nn.Linear(H,H),
torch.nn.Tanh(),
torch.nn.Linear(H,H),
torch.nn.Tanh(),
torch.nn.Linear(H,H),
torch.nn.Tanh(),
torch.nn.Linear(H,H),
torch.nn.Tanh(),
torch.nn.Linear(H,2)
"""
    torch.nn.Linear(1, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, 1),
"""

)