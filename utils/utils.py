import yaml
import torch.distributed as dist
from models.resnet18 import FeatureExtractor
from torch import optim
import torch
import cv2
import numpy as np

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


#def apply_hf_filters(image):
    # Note: The exact values of these filters taken from the paper's figure 2.
    kernel1_values = np.array([
    [-1, 2, -1],
    [2, -4, 2],
    [-1, 2, -1]
    ])
    kernel1 = 1/4 * kernel1_values

# Kernel 2
    kernel2_values = np.array([
    [0, -1, 0],
    [-1, 2, -1],
    [0, -1, 0]
])
    kernel2 = 1/2 * kernel2_values

# Kernel 3
    kernel3_values = np.array([
    [-1, 2, -2, 2, -1],
    [2, -6, 8, -6, 2],
    [-2, 8, -12, 8, -2],
    [2, -6, 8, -6, 2],
    [-1, 2, -2, 2, -1]
])
    kernel3 = 1/12 * kernel3_values


    filtered1 = cv2.filter2D(image, -1, kernel1)
    filtered2 = cv2.filter2D(image, -1, kernel2)
    filtered3 = cv2.filter2D(image, -1, kernel3)
    
    # avg of three filters, need to be adjusted
    hf_image = (filtered1 + filtered2 + filtered3) / 3.0
    return hf_image

#def apply_lf_filters(image):
    # The gaussianblur radaii in paper are (4,6,9) but it throws error ,
    #  we can go with sigma values 4,6,9
    blur1 = cv2.GaussianBlur(image, (3, 3), 0)
    blur2 = cv2.GaussianBlur(image, (5, 5), 0)
    blur3 = cv2.GaussianBlur(image, (7, 7), 0)
    
    #adjust as needed
    lf_image = (blur1 + blur2 + blur3) / 3.0
    return lf_image

def apply_hf_filters(image):
    image_np = np.array(image)
    # Note: The exact values of these filters taken from the paper's figure 2.
    kernel1_values = np.array([
    [-1, 2, -1],
    [2, -4, 2],
    [-1, 2, -1]
    ])
    kernel1 = 1/4 * kernel1_values

# Kernel 2
    kernel2_values = np.array([
    [0, -1, 0],
    [-1, 2, -1],
    [0, -1, 0]
])
    kernel2 = 1/2 * kernel2_values

# Kernel 3
    kernel3_values = np.array([
    [-1, 2, -2, 2, -1],
    [2, -6, 8, -6, 2],
    [-2, 8, -12, 8, -2],
    [2, -6, 8, -6, 2],
    [-1, 2, -2, 2, -1]
])
    kernel3 = 1/12 * kernel3_values


    filtered1 = cv2.filter2D(image_np, -1, kernel1)
    filtered2 = cv2.filter2D(image_np, -1, kernel2)
    filtered3 = cv2.filter2D(image_np, -1, kernel3)
    
    # avg of three filters, need to be adjusted
    hf_image = (filtered1 + filtered2 + filtered3) / 3.0
    return hf_image

def apply_lf_filters(image):
    # Convert image to a NumPy array (assuming it's not already)
    image_np = np.array(image)

    # Apply filtering only to grayscale images
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image_np

    # Apply Gaussian blur
    blur1 = cv2.GaussianBlur(image_gray, (3, 3), 0)
    blur2 = cv2.GaussianBlur(image_gray, (5, 5), 0)
    blur3 = cv2.GaussianBlur(image_gray, (7, 7), 0)

    # Combine blurred images and adjust as needed
    lf_image = (blur1 + blur2 + blur3) / 3.0
    return lf_image