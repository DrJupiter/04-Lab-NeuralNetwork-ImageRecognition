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


# Number of iterations
T = 100
# Allocate space for loss
Loss = np.zeros(T)

for t in range(T):
    # Definer modellens forudsagte y-værdier
    y_pred = model(x)

    # Compute and save loss.
    loss = loss_fn(y_pred, y)
    Loss[t] = loss.item()

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()    


)
