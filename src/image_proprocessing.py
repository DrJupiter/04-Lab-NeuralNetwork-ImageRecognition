import numpy as np
from imageio import imread, imwrite
from skimage.transform import rescale
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray

import glob
from datetime import datetime
import os

def convert_images(input_dir,output_dir):
    files = glob.glob(f"{input_dir}/*")
    for file in files:
        image_raw = imread(f'{file}')
        new_img = rgb2gray(image_raw[:, :, :3])
        new_img = rescale(new_img, 256/image_raw.shape[0], mode='reflect', multichannel=True, anti_aliasing=True)   
        imwrite(f"{output_dir}/{os.path.basename(file)}.png", new_img)


