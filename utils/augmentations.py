#!/usr/bin/python
# _____________________________________________________________________________
import math
# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import sys, os
from typing import Optional, List, Tuple

import torch
from PIL import ImageFilter, ImageOps
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

from torchvision.transforms.v2 import Compose

from utils.constants import DATASETS
from kornia import augmentation as TF
import torchvision


class GaussianBlur(object):
    """
    https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/augmentations.py
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    """
    https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/augmentations.py
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def get_transform_list(args, crop_size=None, tensor_normalize=True, normalize=None):
    transformations = []
    if args.min_crop != 1 and not args.one_crop and not args.crop_first:
        transformations.append(get_resized_crop(args, crop_size))
    if args.flip:
        transformations.append(get_flip(args))
    if args.jitter != 0 and not args.unijit:
        transformations.append(get_jitter(args))
    if args.grayscale and not args.unijit:
        transformations.append(get_grayscale(args))
    if args.blur:
        transformations.append(TF.RandomGaussianBlur(kernel_size=args.blur, sigma=(0.1, 2.0), p=args.pblur))
    if args.solarize:
        transformations.append(TF.RandomSolarize(p=args.solarize))
    if tensor_normalize:
        transformations.append(normalize)
    return torch.nn.Sequential(*transformations)

def get_transformations(args, crop_size=None, tensor_normalize=True):
    norm_dataset = args.dataset
    normalize = TF.Normalize(mean=DATASETS[norm_dataset]['rgb_mean'], std=DATASETS[norm_dataset]['rgb_std'])
    val_transform = normalize

    if args.contrast != 'time' and args.kornia:
        train_transform = get_transform_list(args, crop_size=crop_size, tensor_normalize=tensor_normalize, normalize=normalize)
    else:
        train_transform = val_transform
    return train_transform, val_transform

def get_resized_crop(args, crop_size):
    ratio = (3.0 / 4, 4 / 3.0)
    crop_size = crop_size
    fn = TF.RandomResizedCrop
    return fn(size=crop_size, scale=(args.min_crop, args.max_crop), ratio=ratio, p=args.pcrop)


def get_jitter(args):
    s = args.jitter_strength
    return TF.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s,p=args.jitter)


def get_grayscale(args):
    return TF.RandomGrayscale(p=0.2)


def get_flip(args):
    return TF.RandomHorizontalFlip(p=args.flip)


def get_action_size(args):
    action_size = DATASETS[args.dataset]["action_size"]
    if not args.co3d_quaternion and args.dataset == "CO3D":
        action_size = 14

    return action_size
