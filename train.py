from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchsummary import summary
import numpy as np

from model import combinedModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    model = combinedModel()#.to(device)
    summary(model, (3, 128, 128), device="cpu")

    x = torch.randn((8, 3, 128, 128)).to(device)
    out = model(x)


if __name__ == '__main__':
    main()
