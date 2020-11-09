import numpy as np
from imageio import imread, imwrite
from skimage.transform import rescale, resize
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
     #   new_img = rgb2gray(image_raw[:, :, :3])
        
        image_width = 256
        new_img = rescale(image_raw, (image_width/image_raw.shape[1]), mode='reflect', multichannel=True, anti_aliasing=True)
        #image_width = 256
        #new_img = rescale(new_img, (image_width/image_raw.shape[0], image_width/image_raw.shape[1]), mode='reflect', multichannel=True, anti_aliasing=True)   
        new_img = rgb2gray(new_img[:, :, :3])
        print(new_img.shape)
        imwrite(f"{output_dir}/{os.path.basename(file)}.png", new_img)
