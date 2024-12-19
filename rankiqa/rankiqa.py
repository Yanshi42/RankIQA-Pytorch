import collections
import numpy as np

import torch
from torch import nn
import torchvision

from .preprocessor import Preprocessor


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()

        model = torchvision.models.vgg16(pretrained=False, num_classes=1)
        self.features = torch.nn.Sequential(
            collections.OrderedDict(
                zip(
                    [
                        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
                        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'
                    ],
                    model.features
                )
            )
        )

        self.classifier = torch.nn.Sequential(
            collections.OrderedDict(
                zip(
                    ['fc6_m', 'relu6_m', 'drop6_m', 'fc7_m', 'relu7_m', 'drop7_m', 'fc8_m'],
                    model.classifier
                )
            )
        )

        # Ensure that all parameters in the features part require gradients
        for param in self.features.parameters():
            param.requires_grad = True

    def load_model(self, file, debug: bool = False):
        """
        Load model file.

        :param file: the model file to load.
        :param debug: indicate if output the debug info.
        """
        state_dict = torch.load(file, map_location='cpu')

        dict_to_load = dict()
        for k, v in state_dict.items():  # "v" is parameter and "k" is its name
            for l, p in self.named_parameters():  # "p" is parameter and "l" is its name
                # Use parameter's name to match state_dict's params and model's params
                split_k, split_l = k.split('.'), l.split('.')
                if (split_k[0] in split_l[1]) and (split_k[1] == split_l[2]):
                    dict_to_load[l] = torch.from_numpy(np.array(v)).view_as(p)
                    if debug:  # output debug info
                        print(f"match: {split_k} and {split_l}.")
        
        self.load_state_dict(dict_to_load)

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, start_dim=1, end_dim=-1)  # Flatten for feeding into the classifier
        out = self.classifier(out)
        return out


class RankIQA(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.vgg16 = Vgg16()
        # self.vgg16.load_model("/home/color/myRankIQA/pretrain/Rank_live.caffemodel.pt")

        self.gpu = opt.gpu
        self.eval()
        if self.gpu:
            self.cuda()

        self.preprocessor = Preprocessor(patch_num=30)

    # @torch.no_grad()
    def forward(self, x1, x2):
        # for param in self.vgg16.parameters():
        #     print(param.requires_grad)

        x1 = self.preprocessor(x1)  # Preprocess image -> patches
        if self.gpu: x1 = x1.cuda()
        feat1 = self.vgg16(x1)  # Feature extraction from VGG16

        x2 = self.preprocessor(x2)  # Preprocess image -> patches
        if self.gpu: x2 = x2.cuda()
        feat2 = self.vgg16(x2)  # Feature extraction from VGG16

        return feat1, feat2

    def ranking_loss(self, feat1, feat2, label, margin=1.0):
        # Hinge Loss with margin
        # print(feat1.requires_grad)  # Check if feat1 requires gradient
        # print(feat2.requires_grad)  # Check if feat2 requires gradient
        loss = torch.mean(torch.clamp(feat2 - feat1 + margin, min=0))
        return loss
