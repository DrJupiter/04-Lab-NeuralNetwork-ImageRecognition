import torch
import numpy as np

"""
y = [fish,not_fish]
y_prime = [% fish, % not_fish]
"""
# sum((y-y_prime)^2), but not divided by n
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

IMG_FLATTEN = 256*256*2

model = torch.nn.Sequential(
torch.nn.Linear(IMG_FLATTEN, H),
torch.nn.Tanh(),
torch.nn.Linear(H, H),
torch.nn.Tanh(),
torch.nn.Linear(H, H),
torch.nn.Tanh(),
torch.nn.Linear(H, H),
torch.nn.Tanh(),
torch.nn.Linear(H, H),
torch.nn.Tanh(),
torch.nn.Linear(H, H),
torch.nn.Tanh(),
torch.nn.Linear(H, 2)
)

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Number of iterations
T = 500

# Allocate space for loss
Loss = np.zeros(T)


