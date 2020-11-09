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
    files = glob.glob(f"{input_dir}/*.jpg")
    for key, file in enumerate(files):
        image_raw = imread(f'{file}')
     #   new_img = rgb2gray(image_raw[:, :, :3])
        if len(image_raw.shape) > 3:
            image_raw = image_raw[:, :, :3]
        
        print(image_raw.shape, f"File number {key} out of {len(files)}")
        image_width = 256
        try:
            new_img = rescale(image_raw, (image_width/image_raw.shape[0], image_width/image_raw.shape[1]), mode='reflect', multichannel=True, anti_aliasing=True)   
            new_img = rgb2gray(new_img[:, :, :3])
            imwrite(f"{output_dir}\\fish{key}.png", new_img)
        except:
            pass
        


#convert_images("D:\Pictures\lab4 test\QUT_fish_data\QUT_fish_data\images\\raw_images" , "D:\Pictures\lab4 test\Out")

import re

fish_re = re.compile(r"fish")

def new_img_label(files, index):
    file = files[index]
    img = imread(f'{file}')
    if file.find("fish") > 0:
        label = np.array([1,0])
    else:
        label = np.array([0,1])
    return img, label
