import yaml
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
cfg = read_cfg(cfg_file='/home/air/Spoof/patch_light/config/config.yaml')


import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from Load_FAS_MultiModal_DropModal import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout

# Load an example image
img = Image.open('/home/air/Spoof/Implementation-patchnet/images/LCC_FASD/LCC_FASD_development/spoof/FT720P_G780_REDMI4X_id0_s0_15.png')

# Define the transforms
transform = transforms.Compose([
    transforms.Resize(cfg['model']['image_size']),
    transforms.GaussianBlur(kernel_size=20, sigma=(0.1, 2.0)),
    # transforms.RandomCrop(cfg['dataset']['augmentation']['rand_crop_size']),
    # transforms.RandomHorizontalFlip(
    #     cfg['dataset']['augmentation']['rand_hori_flip']),
    # transforms.RandomRotation(cfg['dataset']['augmentation']['rand_rotation']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])

# Apply the transforms to the image
img_transformed = transform(img)

# Resize the original image to the same size as the transformed image
resize = transforms.Resize(img_transformed.shape[-2:])
img_resized = resize(img)

# Create a grid of images before and after applying the transforms
grid = vutils.make_grid(torch.stack(
    [transforms.ToTensor()(img_resized), img_transformed]), nrow=2)

# Display the grid
import matplotlib.pyplot as plt
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.show()