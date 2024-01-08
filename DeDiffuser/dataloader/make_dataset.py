from PIL import Image
import torch
import cv2
import numpy as np
import os
import torch.utils.data as Data
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from pycocotools.coco import COCO

class MSCOCODataSet(Data.Dataset):
     def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.target_transform = target_transform


if __name__ == '__main__':
    # Get the data and process it
    pass