from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchsummary import summary
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class combinedModel(nn.Module):
    def __init__(self, input_size=(3, 128, 128), output_size=(3, 64, 64)):
        super(combinedModel, self).__init__()
        self.encoder = self.generate_CNN_encoder(input_size)
        self.decoder = self.generate_CNN_decoder(output_size)
        self.pose_head = self.generate_pnp(output_size)

    def forward(self, x):
        bottleneck = self.encoder(x)
        encoding = self.decoder(bottleneck)
        pose = self.pose_head(encoding)

        return bottleneck, encoding, pose

    def generate_ViT_encoder(self, input_size):
        vit_b_16 = models.vit_b_16(pretrained=True)
        #vit_b_32 = models.vit_b_32(pretrained=True)
        #vit_l_16 = models.vit_l_16(pretrained=True)
        #vit_l_32 = models.vit_l_32(pretrained=True)

        return vit_b_16

    def generate_ViT_decoder(self, output_size):

        # some stuff to do

        pass

    def generate_CNN_encoder(self, input_size):
        #input_size = input_size.cuda()
        rnxt50 = models.resnext50_32x4d(pretrained=True)
        #print(rnxt50)
        idx = 0
        for name, param in rnxt50.named_parameters():
            print(idx, ':', name, param.size())
            idx += 1
        #summary(rnxt50, input_size.to(device))

        return rnxt50

    def generate_CNN_decoder(self, output_size):
        up1 = nn.ConvTranspose2d()

        return rnxt50
