### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union, Callable
import random

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
import torchio as tio

### Internal Imports ###
from augmentation import aug
from preprocessing import preprocessing_volumetric as pre_vol
from helpers import utils as u

########################



def mvseg_final_transforms():
    random_flip = tio.RandomFlip(axes=(0, 1), flip_probability=0.5)
    random_gamma = tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5)
    random_affine = tio.RandomAffine(scales=(0.7, 1.2), degrees=10, translation=10, p=0.5)
    random_anisotropy = tio.RandomAnisotropy(downsampling=(1.2, 2.0), p=0.5)
    random_noise = tio.RandomNoise(std=(0, 0.02), p=0.5)
    random_blur = tio.RandomBlur(std=(0, 0.5), p=0.5)
    
    transform_dict = {
        random_flip : 1,
        random_gamma : 1,
        random_affine : 1,
        random_anisotropy : 1,
        random_noise : 1,
        random_blur : 1,
    }
    transform_1 = tio.OneOf(transform_dict)
    transform_2 = tio.OneOf(transform_dict)
    transform_3 = tio.OneOf(transform_dict)
    transform_4 = tio.OneOf(transform_dict)
    transform_5 = tio.OneOf(transform_dict)
    transform_6 = tio.OneOf(transform_dict)
    transforms = tio.Compose([transform_1, transform_2, transform_3, transform_4, transform_5, transform_6])
    return transforms




def mvseg_final_transforms_2():
    random_flip = tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5)
    random_gamma = tio.RandomGamma(log_gamma=(-0.35, 0.35), p=0.5)
    random_affine = tio.RandomAffine(scales=(0.7, 1.3), degrees=10, translation=20, p=0.5)
    random_anisotropy = tio.RandomAnisotropy(downsampling=(1.1, 2.5), p=0.5)
    random_noise = tio.RandomNoise(std=(0, 0.04), p=0.5)
    random_blur = tio.RandomBlur(std=(0, 0.5), p=0.5)
    
    transform_dict = {
        random_flip : 1,
        random_gamma : 1,
        random_affine : 1,
        random_anisotropy : 1,
        random_noise : 1,
        random_blur : 1,
    }
    transform_1 = tio.OneOf(transform_dict)
    transform_2 = tio.OneOf(transform_dict)
    transform_3 = tio.OneOf(transform_dict)
    transform_4 = tio.OneOf(transform_dict)
    transform_5 = tio.OneOf(transform_dict)
    transform_6 = tio.OneOf(transform_dict)
    transforms = tio.Compose([transform_1, transform_2, transform_3, transform_4, transform_5, transform_6])
    return transforms












