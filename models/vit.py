import torch
from torch import nn
from torchvision import models

class FeatureExtractor(nn.Module):
    
    def __init__(self, pretrained=True, device='cpu'):
        super(FeatureExtractor, self).__init__()
        if pretrained:
            self.base_model = models.vit_b_16(pretrained=True)
        else:
            self.base_model = models.vit_b_16(pretrained=False)
        # Remove the final classifier layer
        self.base_model.head = nn.Identity()
        self.device = device
        
    def forward(self, x):
        x = x.to(self.device)
        # The output is already in the (batch_size, seq_length, hidden_dim) format
        out = self.base_model(x)
        return out