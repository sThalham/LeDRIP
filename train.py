from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np

from model import combinedModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    model = combinedModel().to(device)


if __name__ == '__main__':
    main()
