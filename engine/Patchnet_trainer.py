import os
from random import randint
import torch
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from lightning import LightningModule

class PatchModel(LightningModule):
    def __init__(self ,cfg, network, optimizer, loss,loss1, lr_scheduler):
        super().__init__()
        self.network = network.to(torch.device('cuda:0'))
        self.cfg = cfg
        self.network = network
        self.optimizer = optimizer
        self.loss = loss
        self.loss1 = loss1
        self.lr_scheduler = lr_scheduler    
    
    def forward(self,x,y ):
        return self.network(x) , self.network(y)
    
    def _step(self, batch, batch_idx, name, training_step=False):
        # x1 , x2 , dist = batch
        # # x1 , x2 , dist = x1, x2 , dist
        # x1 , x2 = self.forward(x1,x2)
        # self.optimizer.zero_grad()
        # loss = self.loss(x1, x2, dist)
        # loss.backward()
        # self.optimizer.step()
        # score1 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(x1.squeeze()), dim=1)
        # score2 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(x2.squeeze()), dim=1)
        # acc1 = self.calc_acc(score1, dist.squeeze().type(torch.int32))
        # acc2 = self.calc_acc(score2, dist.squeeze().type(torch.int32))
        # accuracy = (acc1 + acc2) / 2
        # pred_label = torch.argmax(score1, dim=1)
        # return loss.item() ,accuracy

        x1, x2, dist = batch
        x1, x2 = self.forward(x1, x2)
        loss = self.loss(x1, x2, dist)
        score1 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(x1.squeeze()), dim=1)
        score2 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(x2.squeeze()), dim=1)
        acc1 = self.calc_acc(score1, dist.squeeze().type(torch.int32))
        acc2 = self.calc_acc(score2, dist.squeeze().type(torch.int32))
        accuracy = (acc1 + acc2) / 2
        pred_label = torch.argmax(score1, dim=1)
        return loss , accuracy

    
    def validation_step(self, batch, batch_idx):
        loss , acc  =  self._step(batch, batch_idx, name="val_loss", training_step=False)
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)

    def training_step(self, batch, batch_idx):
        loss , acc=  self._step(batch, batch_idx, name="train_loss", training_step=True)
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss , acc =  self._step(batch, batch_idx, name="test_loss", training_step=False)
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def calc_acc(self , pred ,target):
        pred = torch.argmax(pred, dim=1)
        equal  =  torch.mean(pred.eq(target).type(torch.FloatTensor))
        return equal.item()
    
    def configure_optimizers(self):
        # opt = torch.optim.SGD(self.parameters(), lr=self.cfg['train']['lr'],
        #                       momentum=self.cfg['train']['momentum'], weight_decay=self.cfg['train']['weight_decay'])
        opt = torch.optim.Adam(self.network.parameters(), lr=self.cfg['train']['lr'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=opt, gamma=self.cfg['train']['lr_decay'])
        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'learning_rate',
            'interval': 'epoch',
            'frequency': 1}
        return [opt], [lr_scheduler]
        # return self.optimizer
    
  