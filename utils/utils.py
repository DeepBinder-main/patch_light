import yaml
import torch.distributed as dist
from models.vit import FeatureExtractor
from torch import optim
import torch

def read_cfg(cfg_file):
    """
    Read configurations from yaml file
    Args:
        cfg_file (.yaml): path to cfg yaml
    Returns:
        (dict): configuration in dict
    """
    with open(cfg_file, 'r') as rf:
        cfg = yaml.safe_load(rf)
        return cfg



def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_optimizer(cfg, network):

    optimizer = None
    if cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=cfg['train']['lr'])
    elif cfg['train']['optimizer'] == 'SGD':
        optimizer = optim.SGD(network.parameters(), lr=cfg['train']['lr'])
    else:
        raise NotImplementedError

    return optimizer


def get_device(cfg):
    device = None
    if cfg['device'] == 'cpu':
        device = torch.device("cpu")
    elif cfg['device'].startswith("cuda"):
        device = torch.device(cfg['device'])
    else:
        raise NotImplementedError
    return device


def build_network(cfg, device):
    network = None

    if cfg['model']['base'] is not None:
        network = FeatureExtractor(pretrained=cfg['model']['pretrained'], device=device)
    else:
        raise NotImplementedError

    return network
