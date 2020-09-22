import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import visdom
viz = visdom.Visdom(env='loss')

from complexYOLO import ComplexYOLO
from kitti import KittiDataset
from region_loss import RegionLoss


batch_size = 4

# dataset
dataset = KittiDataset(root='/home/ding/Documents/deeplearning/kitti',set='train')
data_loader = data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=False)

model = ComplexYOLO()

model.cuda()

# define optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-5 ,momentum = 0.9 , weight_decay = 0.0005)

# define loss function
region_loss = RegionLoss(num_classes=8, num_anchors=5)


lossList = []
totalLossList = []

for epoch in range(200):


   for group in optimizer.param_groups:
       if(epoch>=4 & epoch<80):
           group['lr'] = 1e-4
       if(epoch>=80 & epoch<160):
           group['lr'] = 1e-5
       if(epoch>=160):
           group['lr'] = 1e-6


   losses = 0
   totalLoss = 0
   for batch_idx, (rgb_map, target) in enumerate(data_loader):
          optimizer.zero_grad()

          rgb_map = rgb_map.view(rgb_map.data.size(0),rgb_map.data.size(3),rgb_map.data.size(1),rgb_map.data.size(2))
          output = model(rgb_map.float().cuda())

          loss = region_loss(output,target)
          losses = loss.item()
          totalLoss += loss.item()
          loss.backward()
          optimizer.step()
          lossList.append(losses)

          viz.line(X=np.array(range(epoch * len(data_loader) + batch_idx+1)), Y=np.array(lossList), win='lossBatch',opts={'title':'lossBatch'})

   totalLossList.append(totalLoss / 4 / len(data_loader))
   viz.line(X=np.array(range(epoch + 1)), Y=np.array(totalLossList), win='loss',opts={'title':'loss'})



   if (epoch % 10 == 0):
       torch.save(model, "20181121_"+str(epoch))
