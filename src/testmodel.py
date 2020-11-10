# model-path, test-path, h, out-path 
import torch
from imageio import imread
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import rescale, resize
import glob

import sys

PATH_TO_MODEL = sys.argv[1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMG_FLATTEN = 128*128
H = int(sys.argv[3])


model = torch.nn.Sequential(
torch.nn.Linear(IMG_FLATTEN, IMG_FLATTEN),
torch.nn.ReLU(),
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
torch.nn.ReLU(),
torch.nn.Linear(5, 1)
)

model.to(device)

model.load_state_dict(torch.load(PATH_TO_MODEL))

model.eval()

#img = imread(sys.argv[2])
#data = torch.from_numpy(img.flatten())
#data = data.type(torch.FloatTensor).to(device)
#result = model(data)
#print(result[0], result[1])
#print(result[0] > result[1], result[0] == result[1])
#print(result)

results = []


def test_images(input_dir):
    files = glob.glob(f"{input_dir}/*")
    failed = 0
    for key, file in enumerate(files):
        image_raw = imread(f'{file}')
        if len(image_raw.shape) > 3:
            image_raw = image_raw[:, :, :3]
        
        print(image_raw.shape, f"File number {key} out of {len(files)}: {key/len(files) * 100}%")
        image_width = 128
        new_img = rescale(image_raw, (image_width/image_raw.shape[0], image_width/image_raw.shape[1]), mode='reflect', multichannel=True, anti_aliasing=True)   
        new_img = rgb2gray(new_img[:, :, :3])
        data = torch.from_numpy(new_img.flatten())
        data = data.type(torch.FloatTensor).to(device)
        result = model(data)
        results.append(result/10000) 
    print(failed)


test_images(sys.argv[2])

np.savetxt(f"{sys.argv[4]}/results.txt", results)
print(failed)
