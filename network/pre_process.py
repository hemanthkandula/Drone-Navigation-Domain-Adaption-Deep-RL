
import numpy as np
from torchvision import transforms
import os
from PIL import Image, ImageOps
import numbers
import torch
from torchvision.transforms import Compose


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

# transform = Compose([ToPILImage(), ToTensor()])


def image_train(resize_size=256):

    return transforms.Compose([
        ResizeImage(resize_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(20),
        # transforms.ColorJitter(),

        # transforms.ToTensor(),
        # get_normilize_transform(dataset.lower()),

    ])



def image_test(resize_size=256):

    return transforms.Compose([
        ResizeImage(resize_size),
        # transforms.ToTensor(),
        # get_normilize_transform(dataset.lower())
    ])

