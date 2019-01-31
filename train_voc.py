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


def get_args():
    parser = argparse.ArgumentParser(
        """DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs""")
    parser.add_argument("--image_size", type=int, default=321, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=4, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--dataset", type=str, default="augmentedvoc", choices=["voc2007", "voc2012", "augmentedvoc"],
                        help="The dataset used")
    parser.add_argument("--data_path", type=str, default="data/VOCdevkit", help="the root folder of dataset")
    parser.add_argument("--pre_trained_model", type=str, default="trained_models/vietnh_trained_deeplab_voc")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True,
                       "collate_fn": custom_collate_fn}

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": custom_collate_fn}

    training_set = VOCDataset(opt.data_path, opt.dataset, opt.image_size)
    training_generator = DataLoader(training_set, **training_params)

    test_set = VOCDataset(opt.data_path, opt.dataset, opt.image_size, is_training=False)
    test_generator = DataLoader(test_set, **test_params)

    model = Deeplab(num_classes=training_set.num_classes + 1)
    model.load_state_dict(torch.load(opt.pre_trained_model))
    log_path = os.path.join(opt.log_path, "{}".format(opt.dataset))
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    writer.add_graph(model, torch.rand(opt.batch_size, 3, opt.image_size, opt.image_size))
    if torch.cuda.is_available():
        model.cuda()

    best_loss = 1e10
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epoches):
        for iter, batch in enumerate(training_generator):
            current_step = epoch * num_iter_per_epoch + iter
            current_lr = update_lr(opt.lr, current_step, num_iter_per_epoch * opt.num_epoches)
            optimizer = get_optimizer(model, current_lr, opt.momentum, opt.decay)
            if torch.cuda.is_available():
                batch = [torch.Tensor(record).cuda() for record in batch]
            else:
                batch = [torch.Tensor(record) for record in batch]
            image, gt1, gt2 = batch
            gt1 = gt1.long()
            gt2 = gt2.long()
            optimizer.zero_grad()
            results = model(image)

            mul_losses = multiple_losses(results, [gt1, gt1, gt2, gt1])
            mul_losses[4].backward()
            optimizer.step()
            print(
                "Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {:.2f} (1xloss: {:.2f} 0.75xloss: {:.2f} 0.5xloss: {:.2f} Max_merged_loss: {:.2f})".format(
                    epoch + 1,
                    opt.num_epoches,
                    iter + 1,
                    num_iter_per_epoch,
                    optimizer.param_groups[0]['lr'],
                    mul_losses[4],
                    mul_losses[0],
                    mul_losses[1],
                    mul_losses[2],
                    mul_losses[3]))
            writer.add_scalar('Train/Total_loss', mul_losses[4], current_step)
            writer.add_scalar('Train/1x_scale_loss', mul_losses[0], current_step)
            writer.add_scalar('Train/0.75x_scale_loss', mul_losses[1], current_step)
            writer.add_scalar('Train/0.5x_scale_loss', mul_losses[2], current_step)
            writer.add_scalar('Train/Max_merged_loss', mul_losses[3], current_step)

        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            loss_scale_1_ls = []
            loss_scale_2_ls = []
            loss_scale_3_ls = []
            loss_max_merged_ls = []

            for te_batch in test_generator:
                if torch.cuda.is_available():
                    te_batch = [torch.Tensor(record).cuda() for record in te_batch]
                else:
                    te_batch = [torch.Tensor(record) for record in te_batch]
                te_image, te_gt1, te_gt2 = te_batch
                te_gt1 = te_gt1.long()
                te_gt2 = te_gt2.long()
                num_sample = len(te_gt1)

                with torch.no_grad():
                    te_results = model(te_image)
                    te_mul_losses = multiple_losses(te_results, [te_gt1, te_gt1, te_gt2, te_gt1])
                loss_ls.append(te_mul_losses[4] * num_sample)
                loss_scale_1_ls.append(te_mul_losses[0] * num_sample)
                loss_scale_2_ls.append(te_mul_losses[1] * num_sample)
                loss_scale_3_ls.append(te_mul_losses[2] * num_sample)
                loss_max_merged_ls.append(te_mul_losses[3] * num_sample)

            te_loss = sum(loss_ls) / test_set.__len__()
            te_scale_1_loss = sum(loss_scale_1_ls) / test_set.__len__()
            te_scale_2_loss = sum(loss_scale_2_ls) / test_set.__len__()
            te_scale_3_loss = sum(loss_scale_3_ls) / test_set.__len__()
            te_max_merged_loss = sum(loss_max_merged_ls) / test_set.__len__()

            print(
                "Epoch: {}/{}, Lr: {}, Loss: {:.2f} (1xloss: {:.2f} 0.75xloss: {:.2f} 0.5xloss: {:.2f} Max_merged_loss: {:.2f})".format(
                    epoch + 1,
                    opt.num_epoches,
                    optimizer.param_groups[0]['lr'],
                    te_loss,
                    te_scale_1_loss,
                    te_scale_2_loss,
                    te_scale_3_loss,
                    te_max_merged_loss))

            writer.add_scalar('Test/Total_loss', te_loss, epoch)
            writer.add_scalar('Test/1x_scale_loss', te_scale_1_loss, epoch)
            writer.add_scalar('Test/0.75x_scale_loss', te_scale_2_loss, epoch)
            writer.add_scalar('Test/0.5x_scale_loss', te_scale_3_loss, epoch)
            writer.add_scalar('Test/Max_merged_loss', te_max_merged_loss, epoch)

            model.train()
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                torch.save(model.state_dict(), opt.saved_path + os.sep + "only_params_trained_deeplab_voc")
                torch.save(model, opt.saved_path + os.sep + "whole_model_trained_deeplab_voc")

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break
    writer.close()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
