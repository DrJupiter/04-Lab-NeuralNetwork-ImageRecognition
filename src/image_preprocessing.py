import numpy as np
from imageio import imread, imwrite
from skimage.transform import rescale, resize
from skimage.color import rgb2gray

import glob
from datetime import datetime
import os

import sys

#name = sys.argv[3]

def convert_images(input_dir,output_dir):
    files = glob.glob(f"{input_dir}/*")
    image_width = 128
    for key, file in enumerate(files):
        try:
            image_raw = imread(f'{file}')
            if len(image_raw.shape) > 3:
                image_raw = image_raw[:, :, :3]
            print("Made it 0")
            new_img = rescale(image_raw, (image_width/image_raw.shape[0], image_width/image_raw.shape[1]), mode='reflect', multichannel=True, anti_aliasing=True)   
            print("Made it 1")
            new_img = rgb2gray(new_img[:, :, :3])
            print("Made it 2")
            imwrite(f"{output_dir}/fish{key}.png", new_img)
            print(image_raw.shape, f"File number {key} out of {len(files)}: {key/len(files) * 100}%")
        except:
            pass
        
#convert_images(f"{sys.argv[1]}", f"{sys.argv[2]}")
convert_images("D:\Downl\Fish data\Fish data" , "D:\Pictures\lab4 test\\test_out")

import re

fish_re = re.compile(r"fish")

def new_img_label(files, index):
    file = files[index]
    print(file)
    img = imread(f'{file}')
    if file.find("fish") > 0:
        label = np.array([1,0])
    else:
        label = np.array([0,1])
    return img, label


def new_img_label2(files, index):
    file = files[index]
    print(file)
    img = imread(f'{file}')
    if file.find("fish") > 0:
        label = np.array([10000])
    else:
        label = np.array([0])
    return img, label

