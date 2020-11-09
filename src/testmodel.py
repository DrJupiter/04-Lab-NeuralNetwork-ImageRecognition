import torch
from imageio import imread

import sys

PATH_TO_MODEL = sys.argv[1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMG_FLATTEN = 256*256
H = int(sys.argv[3])

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

model.to(device)

model.load_state_dict(torch.load(PATH_TO_MODEL))

model.eval()

img = imread(sys.argv[2])
data = torch.from_numpy(img.flatten())
data = data.type(torch.FloatTensor).to(device)

print(model(data))
