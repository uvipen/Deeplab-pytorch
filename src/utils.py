"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
import torch


def custom_collate_fn(batch):
    items = list(zip(*batch))
    items[0] = default_collate(items[0])
    items[1] = default_collate(items[1])
    items[2] = default_collate(items[2])
    return items


def get_trainable_params(layer_list):
    for layer in layer_list:
        for module in layer.modules():
            for param in module.parameters():
                if param.requires_grad:
                    yield param


def multiple_losses(results, gts):
    criterion = nn.CrossEntropyLoss()
    losses = [criterion(result, gt) for result, gt in zip(results, gts)]
    losses.append(sum(losses))
    return losses


def update_lr(initialized_lr, current_step, max_step, power=0.9):
    return initialized_lr * ((1 - float(current_step) / max_step) ** (power))


def get_optimizer(model, lr, momentum, decay):
    return torch.optim.SGD([{'params': get_trainable_params(model.dilated_resnet.trainable_variables[:-4])},
                            {'params': get_trainable_params(model.dilated_resnet.trainable_variables[-4:]),
                             'lr': 10 * lr}], lr=lr, momentum=momentum, weight_decay=decay)
