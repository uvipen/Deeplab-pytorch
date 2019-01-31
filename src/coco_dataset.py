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
import json


class COCODataset(Dataset):
    def __init__(self, root_path="data/COCO", image_size=321, is_training=True):
        self.root_path = root_path
        if is_training:
            self.mode = "train2017"
        else:
            self.mode = "val2017"
        id_list_path = os.path.join(self.root_path, "annotations", "stuff_{}.json".format(self.mode))
        data_file = json.load(open(id_list_path, 'r'))
        self.ids = [image["file_name"] for image in data_file["images"]]
        self.classes = ["banner", "blanket", "branch", "bridge", "building-other", "bush", "cabinet", "cage",
                        "cardboard", "carpet", "ceiling-other", "ceiling-tile", "cloth", "clothes", "clouds", "counter",
                        "cupboard", "curtain", "desk-stuff", "dirt", "door-stuff", "fence", "floor-marble",
                        "floor-other", "floor-stone", "floor-tile", "floor-wood", "flower", "fog", "food-other",
                        "fruit", "furniture-other", "grass", "gravel", "ground-other", "hill", "house", "leaves",
                        "light", "mat", "metal", "mirror-stuff", "moss", "mountain", "mud", "napkin", "net", "paper",
                        "pavement", "pillow", "plant-other", "plastic", "platform", "playingfield", "railing",
                        "railroad", "river", "road", "rock", "roof", "rug", "salad", "sand", "sea", "shelf",
                        "sky-other", "skyscraper", "snow", "solid-other", "stairs", "stone", "straw",
                        "structural-other", "table", "tent", "textile-other", "towel", "tree", "vegetable",
                        "wall-brick", "wall-concrete", "wall-other", "wall-panel", "wall-stone", "wall-tile",
                        "wall-wood", "water-other", "waterdrops", "window-blind", "window-other", "wood", "other"]

        self.class_ids = [92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                          112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
                          131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
                          150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
                          169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183]
        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.num_images = len(self.ids)
        self.is_training = is_training

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        id = self.ids[item]
        image_path = os.path.join(self.root_path, "images", "{}".format(self.mode),
                                  "{}".format(id))
        gt_image_path = os.path.join(self.root_path, "images", "stuff_{}_pixelmaps".format(self.mode),
                                  "{}.png".format(id[:-4]))
        image = cv2.imread(image_path).astype(np.float32)
        image[:, :, 0] -= 104.008
        image[:, :, 1] -= 116.669
        image[:, :, 2] -= 122.675

        gt_image = Image.open(gt_image_path).convert('P')
        gt_image = np.asarray(gt_image, np.int32)
        gt_image[gt_image == 255] = 0
        gt_image[gt_image != 0] -= (len(self.classes) - 1)


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
