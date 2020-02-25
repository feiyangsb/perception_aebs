"""
@author Feiyang Cai
@email feiyang.cai@vanderbilt.edu
@create date 2020-02-25 09:17:43
@modify date 2020-02-25 09:17:43
@desc This is the trainer for the regression module in CARLA AEBS
"""
from scripts.perception_net import PerceptionNet
from scripts.data_loader import CarlaAEBSDataset
from torch.utils.data import DataLoader
import torch
import time
import os
import numpy as np
import logging
import torch.optim as optim

logger_path = "./log"
if not os.path.exists(logger_path):
    os.makedirs(logger_path)
logging.basicConfig(level=logging.INFO, filename=os.path.join(logger_path, 'perception_epoch_350_Feb_25.log'))

class PerceptionTrainer():
    def __init__(self, data_path, epoch):
        self.dataset = CarlaAEBSDataset(data_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = epoch
    
    def fit(self):
        self.model = PerceptionNet()
        self.model = self.model.to(self.device)
        data_loader = DataLoader(self.dataset, batch_size=64, shuffle=True, num_workers=8)
        loss_func = torch.nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.5e-6, amsgrad=False) 
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(self.epoch*0.7)], gamma=0.1)
        self.model.train()

        for epoch in range(self.epoch):
            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for batch_idx, (images, gt_distances) in enumerate(data_loader):
                images = images.to(device=self.device, dtype=torch.float)
                gt_distances = gt_distances.to(self.device, dtype=torch.float)
                optimizer.zero_grad()

                outputs = self.model(images)
                loss = loss_func(outputs, gt_distances)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                n_batches += 1

            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            logging.info('Epoch {}/{}\t Time: {:.3f}\t Total Loss: {:.8f}'\
                        .format(epoch+1, self.epoch, epoch_train_time, loss_epoch/n_batches))
            print('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(epoch+1, self.epoch, epoch_train_time, loss_epoch/n_batches))
        return self.model
    
    def save_model(self, path='./models/'):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), os.path.join(path, "perception.pt"))
