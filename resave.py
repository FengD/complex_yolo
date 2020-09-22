import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

from complexYOLO import ComplexYOLO
from kitti import KittiDataset
from region_loss import RegionLoss


batch_size = 4

# dataset
dataset = KittiDataset(root='/home/ding/Documents/deeplearning/kitti',set='train')
data_loader = data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=False)

# model = ComplexYOLO()
model = torch.load('ComplexYOLO_epochkitti784047')
torch.save(model.state_dict(), "ComplexYOLO_epochkitti784047_value")
