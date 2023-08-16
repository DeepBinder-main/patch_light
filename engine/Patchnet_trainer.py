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
    def __init__(self ,cfg, network, optimizer, loss, lr_scheduler):
        super().__init__()
        self.network = network.to(torch.device('cuda:0'))
        self.cfg = cfg
        self.network = network
        self.optimizer = optimizer
        self.loss = loss
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
        opt = torch.optim.SGD(self.parameters(), lr=self.cfg['train']['lr'],
                              momentum=self.cfg['train']['momentum'], weight_decay=self.cfg['train']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=opt, gamma=self.cfg['train']['lr_decay'])
        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'learning_rate',
            'interval': 'epoch',
            'frequency': 1}
        return [opt], [lr_scheduler]
        # return self.optimizer
    
    # def load_model(self):
    #     saved_name = os.path.join(self.cfg['output_dir'], '{}_{}.pth'.\
    #         format(self.cfg['model']['base'], self.cfg['dataset']['name']))
    #     state = torch.load(saved_name)

    #     self.optimizer.load_state_dict(state['optimizer'])
    #     self.network.load_state_dict(state['state_dict'])
    #     self.loss.load_state_dict(state['loss'])
        
    # def save_model(self, epoch, val_loss):
    #     if not os.path.exists(self.cfg['output_dir']):
    #         os.makedirs(self.cfg['output_dir'])

    #     saved_name = os.path.join(self.cfg['output_dir'], '{}_{}_{}.pth'.\
    #         format(self.cfg['model']['base'], epoch, val_loss))

    #     state = {
    #         'epoch': epoch,
    #         'state_dict': self.network.state_dict(),
    #         'optimizer': self.optimizer.state_dict(),
    #         'loss': self.loss.state_dict()
    #     }
        
    #     torch.save(state, saved_name)
        
    # def train_one_epoch(self, epoch):

    #     self.network.train()
    #     self.train_loss_metric.reset(epoch)
    #     self.train_acc_metric.reset(epoch)

    #     for i, (img1, img2, label) in enumerate(self.trainloader):
    #         img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)
    #         feature1 = self.network(img1)
    #         feature2 = self.network(img2)
    #         self.optimizer.zero_grad()
    #         loss = self.loss(feature1, feature2, label)
    #         loss.backward()
    #         self.optimizer.step()

    #         score1 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(feature1.squeeze()), dim=1)
    #         score2 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(feature2.squeeze()), dim=1)

    #         acc1 = calc_acc(score1, label.squeeze().type(torch.int32))
    #         acc2 = calc_acc(score2, label.squeeze().type(torch.int32))
    #         accuracy = (acc1 + acc2) / 2
            
    #         # Update metrics
    #         self.train_loss_metric.update(loss.item())
    #         self.train_acc_metric.update(accuracy)

    #         print('Epoch: {:3}, iter: {:5}, loss: {:.5}, acc: {:.5}'.\
    #             format(epoch, epoch * len(self.trainloader) + i, \
    #             self.train_loss_metric.avg, self.train_acc_metric.avg))
        
    #     if self.lr_scheduler is not None:
    #         self.lr_scheduler.step()
        
    # def train(self):
    #     for epoch in range(self.cfg['train']['num_epochs']):
    #         self.train_one_epoch(epoch)
    #         epoch_loss = self.validate(epoch)
    #         # if epoch_acc > self.best_val_acc:
    #         #     self.best_val_acc = epoch_acc
    #         self.save_model(epoch, epoch_loss)
            
    # def validate(self, epoch):
    #     self.network.eval()
    #     self.val_loss_metric.reset(epoch)
    #     self.val_acc_metric.reset(epoch)

    #     seed = randint(0, len(self.valloader)-1)
        
    #     with torch.no_grad():
    #         for i, (img1, img2, label) in enumerate(self.valloader):
    #             img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)
    #             feature1 = self.network(img1)
    #             feature2 = self.network(img2)
    #             loss = self.loss(feature1, feature2, label)

    #             score1 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(feature1.squeeze()), dim=1)
    #             score2 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(feature2.squeeze()), dim=1)

    #             acc1 = calc_acc(score1, label.squeeze().type(torch.int32))
    #             acc2 = calc_acc(score2, label.squeeze().type(torch.int32))
    #             accuracy = (acc1 + acc2) / 2

    #             # Update metrics
    #             self.val_loss_metric.update(loss.item())
    #             self.val_acc_metric.update(accuracy)
        
    #     print("Validation epoch {} =============================".format(epoch))
    #     print("Epoch: {:3}, loss: {:.5}, acc: {:.5}".format(epoch, self.val_loss_metric.avg, self.val_acc_metric.avg))
    #     print("=================================================")

    #     return self.val_loss_metric.avg
                
