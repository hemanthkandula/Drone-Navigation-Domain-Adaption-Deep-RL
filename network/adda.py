
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import pdb
import torch.nn.utils.weight_norm as weightNorm


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

########################################################################################################################

class ResNetEncoder(nn.Module):
    """ResNet encoder model for ADDA."""

    def __init__(self):
        """Init ResNet encoder."""
        super(ResNetEncoder, self).__init__()

        self.restored = False

        # model_resnet = resnet_dict[resnet_name](pretrained=True)
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        # self.new_cls = new_cls
        # print("classes inside network",new_cls)

        # self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
        # self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        return x

    def get_parameters(self):

        parameter_list = [{"params": self.feature_layers.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                          # {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}
                          ]

        return parameter_list

class ResNetClassifier(nn.Module):
    """ResNet classifier model for ADDA."""
    def __init__(self,class_num):
        """Init ResNet encoder."""
        super(ResNetClassifier, self).__init__()
        # model_resnet = models.resnet50(pretrained=True)

        self.fc = nn.Linear(2048, class_num)
        self.fc.apply(init_weights)


        # self.__in_features = model_resnet.fc.in_features

    def forward(self, feat):
        """Forward the ResNet classifier."""
        # out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc(feat)
        return out

    def get_parameters(self):

        parameter_list = [
                        # {"params": self.feature_layers.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                          {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}
                          ]

        return parameter_list
########################################################################################################################
