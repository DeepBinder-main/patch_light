import os
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import timm
import torch.backends.cudnn as cudnn
# from datasets.dataset import MyDataset
# from pl_models import *
import json
import sys
import numpy as np
from torchvision import transforms, datasets
from dataset.FAS_dataset import FASDataset
from torch.optim.lr_scheduler import StepLR
from metrics.losses import PatchLoss


ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.utils import read_cfg , get_rank , get_optimizer, build_network, \
    get_device

from utils.utils import read_cfg
cfg = read_cfg(cfg_file='config/config.yaml')

# fix the seed for reproducibility
seed = cfg['seed'] + get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.benchmark = True

# backend = ""
# if not backend == "tf32":
#     torch.backends.cuda.matmul.allow_tf32 = False
#     torch.backends.cudnn.allow_tf32 = False
# else:
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

# build model and engine
device = get_device(cfg)
network = build_network(cfg, device)
network.to(device)
optimizer = get_optimizer(cfg, network)
lr_scheduler = StepLR(optimizer=optimizer, step_size=90, gamma=0.5)
criterion = PatchLoss().to(device=device)


# Without Resize transform, images are of different sizes and it causes an error
train_transform = transforms.Compose([
    
    transforms.Resize(cfg['model']['image_size']),
    transforms.RandomCrop(cfg['dataset']['augmentation']['rand_crop_size']),
    transforms.RandomHorizontalFlip(cfg['dataset']['augmentation']['rand_hori_flip']),
    transforms.RandomRotation(cfg['dataset']['augmentation']['rand_rotation']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])

val_transform = transforms.Compose([
    transforms.Resize(cfg['model']['image_size']),
    transforms.RandomCrop(cfg['dataset']['augmentation']['rand_crop_size']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])

trainset = FASDataset(
    root_dir=cfg['dataset']['root'],
    transform=train_transform,
    csv_file=cfg['dataset']['train_set'],
    is_train=True
)

valset = FASDataset(
    root_dir=cfg['dataset']['root'],
    transform=val_transform,
    csv_file=cfg['dataset']['val_set'],
    is_train=False
)

trainloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=cfg['train']['batch_size'],
    shuffle=True,
    num_workers=4
)

valloader = torch.utils.data.DataLoader(
    dataset=valset,
    batch_size=cfg['val']['batch_size'],
    shuffle=False,
    num_workers=4
)

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from engine.Patchnet_trainer import PatchModel
import lightning as L

model = PatchModel(cfg=cfg, network=network, optimizer=optimizer, loss=criterion, lr_scheduler=lr_scheduler)

logger = TensorBoardLogger("tb_logs", name="my_model")

from lightning.pytorch.callbacks import ModelCheckpoint

callbacks = [
    ModelCheckpoint(save_top_k=1, mode="max", monitor="val_acc", save_last=True)
]

trainer = L.Trainer(
    callbacks=callbacks,
    max_epochs=100,
    accelerator="gpu",
    devices=[0],
    logger=logger ,
    deterministic=True,

)


trainer.fit(model, trainloader, valloader)

