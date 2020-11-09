import torch
from imageio import imread

import sys

PATH_TO_MODEL = sys.argv[1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMG_FLATTEN = 256*256
H = int(sys.argv[3])


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

model.load_state_dict(torch.load(PATH_TO_MODEL))

model.eval()

img = imread(sys.argv[2])
data = torch.from_numpy(img.flatten())
data = data.type(torch.FloatTensor).to(device)

print(model(data))
