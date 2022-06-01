from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np


class combinedModel(nn.Module):
    def __init__(self, input_size=(128, 128, 3), output_size=(64, 64, 3)):
        super(combinedModel, self).__init__()
        self.encoder = self.generate_encoder(input_size)
        self.decoder = self.generate_decoder(output_size)
        self.pose_head = self.generate_pnp(output_size)

    def forward(self, x):
        bottleneck = self.encoder(x)
        encoding = self.decoder(bottleneck)
        pose = self.pose_head(encoding)

        return bottleneck, encoding, pose

    def generate_encoder(self, input_size):
        vit_b_16 = models.vit_b_16(pretrained=True)
        #vit_b_32 = models.vit_b_32(pretrained=True)
        #vit_l_16 = models.vit_l_16(pretrained=True)
        #vit_l_32 = models.vit_l_32(pretrained=True)

        return vit_b_16
