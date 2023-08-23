import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.utils import read_cfg
cfg = read_cfg(cfg_file='config/config.yaml')

class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return self.mse(input, target)
    

# class AdMSoftmaxLoss(nn.Module):

#     def __init__(self, in_features, out_features, s=30.0, m=0.4):
#         '''
#         AM Softmax Loss
#         '''
#         super(AdMSoftmaxLoss, self).__init__()
#         self.s = s
#         self.m = m
#         self.in_features = in_features
#         self.out_features = out_features
#         self.fc = nn.Linear(in_features, out_features, bias=False)
        
#     def forward(self, x, labels):
#         '''
#         input shape (N, in_features)
#         '''
#         assert len(x) == len(labels)
#         assert torch.min(labels) >= 0
#         assert torch.max(labels) < self.out_features
        
#         # Normalize parameters and input
#         for W in self.fc.parameters():
#             W = F.normalize(W, p=2, dim=1)
#         x = F.normalize(x, p=2, dim=1)

#         wf = self.fc(x)
#         numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
#         excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
#         denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)

#         L = numerator - torch.log(denominator)
        
#         return - torch.mean(L)

class AdMSoftmaxLoss(nn.Module):

    # def __init__(self, in_features, out_features, s=30.0, m=0.4):
    def __init__(self, in_features, out_features, s=30.0, m_l=0.4, m_s=0.1):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = [m_s, m_l]
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        '''
        input: 
            x shape (N, in_features)
            labels shape (N)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W , dim=1)

        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        m = torch.tensor([self.m[ele] for ele in labels]).to(x.device)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        
        L = numerator - torch.log(denominator)
        
        return - torch.mean(L)



class PatchLoss(nn.Module):

    def __init__(self, alpha1=1.0, alpha2=1.0,s=30.0,m_l=0.4,m_s=0.1):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.sim_loss = SimilarityLoss()
        self.amsm_loss = AdMSoftmaxLoss(cfg['model']['out_feat'],2)
        self.s= s
        self.m_l = m_l
        self.m_s = m_s


    
    def forward(self, x1, x2, label):
        amsm_loss1 = self.amsm_loss(x1.squeeze(), label.type(torch.long).squeeze())
        amsm_loss2 = self.amsm_loss(x2.squeeze(), label.type(torch.long).squeeze())
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        sim_loss = self.sim_loss(x1, x2)
        loss = self.alpha1 * sim_loss + self.alpha2 * (amsm_loss1 + amsm_loss2)
        
        return loss



class ArcFaceLoss(nn.Module):

    def __init__(self, in_features, out_features, s=64, m_l=0.5, m_s=0.1):
        '''
        Angular Penalty Softmax Loss

        Four 'loss_types' available: ['arcface', 'sphereface', 'cosface', 'adacos']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        AdaCos: https://arxiv.org/abs/1905.00292

        '''

        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = [m_l,m_s]
        # loss_type = loss_type.lower()
        # assert loss_type in  ['arcface', 'sphereface', 'cosface', 'adacos']
        loss_type  = 'sphereface'
        # if loss_type == 'arcface':
        #     self.s = 64.0 if not s else s
        #     self.m = 0.5 if not m else m
        # elif loss_type == 'sphereface':
        #     self.s = 64.0 if not s else s
        #     self.m = 1.35 if not m else m
        # elif loss_type == 'cosface':
        #     self.s = 30.0 if not s else s
        #     self.m = 0.4 if not m else m
        # elif loss_type == 'adacos':
        #     self.s = math.sqrt(2) * math.log(out_features - 1) if not s else s
        #     self.m = 0.50 if not m else 
        self.s = math.sqrt(2) * math.log(out_features - 1) 
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        m = torch.tensor([self.m[ele] for ele in labels]).to(x.device)
        self.loss_type = 'sphereface'
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        elif self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        elif self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))
        elif self.loss_type == 'adacos':
            theta = torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.0 + 1e-7, 1.0 - 1e-7))
            numerator = torch.cos(theta + self.m)
            one_hot = torch.zeros_like(wf)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
            numerator = wf * (1 - one_hot) + numerator * one_hot
            B_avg = torch.where(one_hot < 1, self.s * torch.exp(wf), torch.zeros_like(wf)).sum(dim=0).mean()
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
            numerator *= self.s

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class PatchLoss1(nn.Module):

    def __init__(self, alpha1=1.0, alpha2=1.0,s=64, m_l=0.5,m_s=0.1):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.sim_loss = SimilarityLoss()
        self.arc_loss = ArcFaceLoss(cfg['model']['out_feat'],2)
        self.s= s
        self.m_l = m_l
        self.m_s = m_s
    
    def forward(self, x1, x2, label):
        amsm_loss1 = self.arc_loss(x1.squeeze(), label.type(torch.long).squeeze())
        amsm_loss2 = self.arc_loss(x2.squeeze(), label.type(torch.long).squeeze())
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        sim_loss = self.sim_loss(x1, x2)
        loss = self.alpha1 * sim_loss + self.alpha2 * (amsm_loss1 + amsm_loss2)
        
        return loss