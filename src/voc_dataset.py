"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
from math import ceil


class VOCDataset(Dataset):
    def __init__(self, root_path="data/VOCdevkit", dataset="voc2012", image_size=321, is_training=True):
        self.dataset = dataset
        if self.dataset == "voc2007":
            self.data_path = os.path.join(root_path, "VOC2007")
            if is_training:
                id_list_path = os.path.join(self.data_path, "ImageSets/Segmentation/trainval.txt")
            else:
                id_list_path = os.path.join(self.data_path, "ImageSets/Segmentation/test.txt")
        elif self.dataset == "voc2012":
            self.data_path = os.path.join(root_path, "VOC2012")
            if is_training:
                id_list_path = os.path.join(self.data_path, "ImageSets/Segmentation/train.txt")
            else:
                id_list_path = os.path.join(self.data_path, "ImageSets/Segmentation/val.txt")
        elif self.dataset == "augmentedvoc":
            self.data_path = os.path.join(root_path, "VOCaugmented")
            if is_training:
                id_list_path = os.path.join(self.data_path, "list/train_aug.txt")
            else:
                id_list_path = os.path.join(self.data_path, "list/val.txt")

        self.ids = [id.strip() for id in open(id_list_path)]
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']
        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.num_images = len(self.ids)
        self.is_training = is_training

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        id = self.ids[item]
        if self.dataset in ["voc2007", "voc2012"]:
            image_path = os.path.join(self.data_path, "JPEGImages", "{}.jpg".format(id))
            gt_image_path = os.path.join(self.data_path, "SegmentationClass", "{}.png".format(id))
        elif self.dataset == "augmentedvoc":
            image_path = os.path.join(self.data_path, "img", "{}.jpg".format(id))
            gt_image_path = os.path.join(self.data_path, "gt", "{}.png".format(id))
        image = cv2.imread(image_path).astype(np.float32)
        image[:, :, 0] -= 104.008
        image[:, :, 1] -= 116.669
        image[:, :, 2] -= 122.675

        gt_image = Image.open(gt_image_path).convert('P')
        gt_image = np.asarray(gt_image, np.int32)
        gt_image[gt_image == 255] = 0

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        gt_image = cv2.resize(gt_image, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        gt_torch = torch.Tensor(torch.from_numpy(gt_image[None, None, :, :]).float())

        gt1_size = ceil(self.image_size / 8.)
        interp = nn.Upsample(size=(gt1_size, gt1_size), mode='bilinear', align_corners=True)
        gt1 = interp(gt_torch).data.numpy()[0, 0, :, :]

        gt2_size = ceil(self.image_size / 16.)
        interp = nn.Upsample(size=(gt2_size, gt2_size), mode='bilinear', align_corners=True)
        gt2 = interp(gt_torch).data.numpy()[0, 0, :, :]

        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(gt1, dtype=np.float32), np.array(
            gt2, dtype=np.float32)
