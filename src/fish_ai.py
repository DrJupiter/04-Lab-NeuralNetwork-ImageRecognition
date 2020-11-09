import torch
import numpy as np
import random
import glob
from image_preprocessing import new_img_label, new_img_label2

import sys


# Training images
input_dir, output_dir = sys.argv[1:3] 
images = glob.glob(f"{input_dir}/*")
N_IMG = len(images)-1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print(device)
"""
y = [fish,not_fish]
y_prime = [% fish, % not_fish]
"""
# sum((y-y_prime)^2), but not divided by n
#Loss_fn = torch.nn.BCELoss(reduction='sum') #mean
Loss_fn = torch.nn.MSELoss(reduction='sum')

H = int(sys.argv[3])

# 10 -> 2
# 256,256, 2

# img
# torch.flatten(img) -> 256*256*2 = 131072
# torch.nn.Linear(131072, H)

#TODO størrelsen af imput vektoren afhænger af fildata, så hvis vi gerne vil bevare størrelsesforholdet  billedet, bliver vi nødt til at lade imputtet
#til vores NN være afhængingt at billdet, medmindre alt vores data har samme dimensioner. Det har jeg ingen anelse om
IMG_FLATTEN = 256*256
"""
model = torch.nn.Sequential(
torch.nn.Linear(IMG_FLATTEN, H),
torch.nn.Sigmoid(),
torch.nn.Linear(H, H),
torch.nn.Sigmoid(),
torch.nn.Linear(H, H),
torch.nn.Sigmoid(),
torch.nn.Linear(H, H),
torch.nn.Sigmoid(),
torch.nn.Linear(H, H),
torch.nn.Sigmoid(),
torch.nn.Linear(H, H),
torch.nn.Sigmoid(),
torch.nn.Linear(H, H),
torch.nn.Sigmoid(),
torch.nn.Linear(H, H),
torch.nn.Sigmoid(),
torch.nn.Linear(H, H),
torch.nn.Sigmoid(),
torch.nn.Linear(H, H),
torch.nn.Sigmoid(),
torch.nn.Linear(H, H),
torch.nn.Sigmoid(),
torch.nn.Linear(H, H),
torch.nn.Sigmoid(),
torch.nn.Linear(H, H),
torch.nn.Sigmoid(),
torch.nn.Linear(H, H),
torch.nn.Sigmoid(),
torch.nn.Linear(H, H),
torch.nn.Sigmoid(),
torch.nn.Linear(H, H),
torch.nn.Sigmoid(),
torch.nn.Linear(H, H),
torch.nn.Sigmoid(),
torch.nn.Linear(H, 2)
)
"""
X = [int(H/2**x) for x in range(0,11)]

X = np.array(X)

model = torch.nn.Sequential(
torch.nn.Linear(IMG_FLATTEN, H),
torch.nn.ReLU(),
torch.nn.Linear(H, H),
torch.nn.ReLU(),
torch.nn.Linear(H, 3000),
torch.nn.ReLU(),
torch.nn.Linear(3000, 3000),
torch.nn.ReLU(),
torch.nn.Linear(3000, 1500),
torch.nn.ReLU(),
torch.nn.Linear(1500, 1500),
torch.nn.ReLU(),
torch.nn.Linear(1500, 750),
torch.nn.ReLU(),
torch.nn.Linear(750, 375),
torch.nn.ReLU(),
torch.nn.Linear(375, 187),
torch.nn.ReLU(),
torch.nn.Linear(187, 93),
torch.nn.ReLU(),
torch.nn.Linear(93, 46),
torch.nn.ReLU(),
torch.nn.Linear(46, 23),
torch.nn.ReLU(),
torch.nn.Linear(23, 11),
torch.nn.ReLU(),
torch.nn.Linear(11, 5),
torch.nn.Sigmoid(),
torch.nn.Linear(5, 2)
)


model.to(device)



learning_rate = float(sys.argv[4])
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Number of iterations
T = int(sys.argv[5])

# Allocate space for loss
Loss = np.zeros(T)

#Initialize generator
random.seed()


for t in range(T):

    img, label = new_img_label(images, random.randint(0,N_IMG)) 
    label = torch.tensor(label, dtype=torch.float32).to(device)

    # Definer modellens forudsagte y-værdier
#    print((torch.flatten(torch.from_numpy(img))).size())
    data = torch.from_numpy(img.flatten())
    data = data.type(torch.FloatTensor).to(device)
    y_pred = model(data)

    # Compute and save loss.
    loss = Loss_fn(y_pred, label)
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

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

print(Loss)
np.savetxt(f"{output_dir}/loss.txt", Loss, delimiter=",")

#with open(f"{output_dir}/loss.txt", "w") as txt_file:
#    for line in Loss:
#        txt_file.write(" ".join(line) + "\n")

torch.save(model.state_dict(), f"{output_dir}/Result")
