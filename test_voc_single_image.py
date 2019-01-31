"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from src.voc_dataset import VOCDataset
from src.utils import custom_collate_fn, multiple_losses, update_lr, get_optimizer
from src.deeplab import Deeplab
from tensorboardX import SummaryWriter
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image
import glob


def get_args():
    parser = argparse.ArgumentParser(
        """DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs""")
    parser.add_argument("--image_size", type=int, default=321, help="The common width and height for all images")
    parser.add_argument("--test_set", type=str, default="val",
                        help="For both VOC2007 and 2012, you could choose 3 different datasets: train, trainval and val. Additionally, for VOC2007, you could also pick the dataset name test")
    parser.add_argument("--year", type=str, default="2012", help="The year of dataset (2007 or 2012)")
    parser.add_argument("--data_path", type=str, default="data/VOCdevkit", help="the root folder of dataset")
    parser.add_argument("--pre_trained_model_type", type=str, choices=["model", "params"], default="model")
    parser.add_argument("--pre_trained_model_path", type=str, default="trained_models/whole_model_trained_deeplab_voc")
    parser.add_argument("--output", type=str, default="predictions")

    args = parser.parse_args()
    return args

full_to_colour = {0: (0, 0, 0), 1: (128, 64, 128), 2: (244, 35, 232), 3: (70, 70, 70), 4: (102, 102, 156),
                  5: (190, 153, 153), 6: (153, 153, 153), 7: (250, 170, 30), 8: (220, 220, 0), 9: (107, 142, 35),
                  10: (152, 251, 152), 12: (70, 130, 180), 13: (220, 20, 60), 14: (255, 0, 0), 15: (0, 0, 142),
                  16: (0, 0, 70), 17: (0, 60, 100), 18: (0, 80, 100), 19: (0, 0, 230), 20: (119, 11, 32)}

def test(opt):
    output_folder = os.path.join(opt.output, "VOC{}_{}".format(opt.year, opt.test_set))
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    if torch.cuda.is_available():
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path)
        else:
            model = Deeplab(num_classes=21)
            model.load_state_dict(torch.load(opt.pre_trained_model_path))
    else:
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage)
        else:
            model = Deeplab(num_classes=21)
            model.load_state_dict(torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage))

    model.eval()
    for image_path in glob.iglob("test_images/" + '*.jpg'):
        if "prediction" in image_path:
            continue
        img = np.zeros((513, 513, 3))
        image = cv2.imread(image_path).astype(np.float32)
        h,w,_ = image.shape
        image = cv2.resize(image, (int(w/4), int(h/4)))
        h1, w1,_ = image.shape
        image[:, :, 0] -= 104.008
        image[:, :, 1] -= 116.669
        image[:, :, 2] -= 122.675
        img[:h1, :w1, :] = image

        img = np.transpose(np.array(img, dtype=np.float32), (2, 0, 1))
        img = img[None, :, :, :]
        img = torch.Tensor(img)
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            result = model(img)
        interp = nn.UpsamplingBilinear2d(size=(513, 513))
        output = interp(result[3]).cpu().data[0].numpy()
        output = output[:, :h1, :w1]
        result = np.argmax(output.transpose(1,2,0), axis=2).astype(np.uint8)

        pred_colour = np.zeros((3, result.shape[0], result.shape[1]))
        for k, v in full_to_colour.items():
            pred_r = np.zeros((result.shape[0], result.shape[1]))
            pred_r[(result == k)] = v[0]
            pred_g = np.zeros((result.shape[0], result.shape[1]))
            pred_g[(result == k)] = v[1]
            pred_b = np.zeros((result.shape[0], result.shape[1]))
            pred_b[(result == k)] = v[2]
            uuu = np.stack((pred_r, pred_g, pred_b))
            pred_colour += uuu
        save_image(torch.from_numpy(pred_colour).float().div(255), image_path[:-4] + "_prediction.jpg")



if __name__ == "__main__":
    opt = get_args()
    test(opt)
