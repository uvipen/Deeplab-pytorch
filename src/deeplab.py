"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch.nn as nn
import torch
from math import ceil


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                for p in m.parameters():
                    p.requires_grad = False
            # if isinstance(m, nn.BatchNorm2d):
            #     for p in m.parameters():
            #         p.requires_grad = False

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DilatedResNet(nn.Module):
    def __init__(self, block, model="resnet101", num_classes=1000):
        super(DilatedResNet, self).__init__()
        if model == "resnet50":
            layers = [3, 4, 6, 3]
        else:
            layers = [3, 4, 23, 3]  # resnet101
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5_1 = nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=6, dilation=6, bias=True)
        self.layer5_2 = nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.layer5_3 = nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=18, dilation=18, bias=True)
        self.layer5_4 = nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=24, dilation=24, bias=True)

        self._initialize_weights()
        self.trainable_variables = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4,
                                    self.layer5_1, self.layer5_2, self.layer5_3, self.layer5_4]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                for p in m.parameters():
                    p.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride, dilation):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        out = self.layer5_1(x)
        out += self.layer5_2(x)
        out += self.layer5_3(x)
        out += self.layer5_4(x)

        return out


class Deeplab(nn.Module):
    def __init__(self, block=Bottleneck, num_classes=21):
        super(Deeplab, self).__init__()
        self.dilated_resnet = DilatedResNet(block=block, num_classes=num_classes)

    def forward(self, x):
        input_size = x.size()[2:4]
        self.upsampl1_1 = nn.Upsample(size=(int(input_size[0] * 0.75) + 1, int(input_size[1] * 0.75) + 1),
                                      mode='bilinear', align_corners=True)
        self.upsampl1_2 = nn.Upsample(size=(int(input_size[0] * 0.5) + 1, int(input_size[1] * 0.5) + 1),
                                      mode='bilinear', align_corners=True)
        self.upsampl2 = nn.Upsample(size=(int(input_size[0] * 0.125) + 1, int(input_size[1] * 0.125) + 1),
                                    mode='bilinear', align_corners=True)

        x1 = self.dilated_resnet(x)

        x2 = self.upsampl1_1(x)
        x2 = self.dilated_resnet(x2)
        x2 = self.upsampl2(x2)

        x3 = self.upsampl1_2(x)
        x3 = self.dilated_resnet(x3)

        x4 = torch.max(torch.max(x1, x2), self.upsampl2(x3))

        return [x1, x2, x3, x4]
