import torch
import torch.nn as nn
import config as config
#from dice import DiceLoss
#from DiceLoss import DiceLoss
import torch.nn.functional as F
#from torchgeometry.losses.dice import DiceLoss
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss,self).__init__()
    def _dice_loss(self,inputs,targets,eps=1e-7):
        targets=targets.type(torch.int64)
        true_1_hot=torch.eye(2,device="cuda")[targets.to("cuda").squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        inputs=torch.sigmoid(inputs)
        ninputs=1-inputs
        provas=torch.cat([inputs,ninputs],dim=1)
        true_1_hot=true_1_hot.type(inputs.type())
        dims=(0,)+tuple(range(2,targets.ndimension()))
        intersection=torch.sum(provas*true_1_hot,dims)
        cardinality=torch.sum(provas+true_1_hot,dims)
        diceloss=(2.*intersection/(cardinality+eps)).mean()
        loss=(1-diceloss)
        return loss
    def forward(self,preds,mask,label,intensity):
        crossEntropy = nn.CrossEntropyLoss()
        binaryCrossEntropy = nn.BCEWithLogitsLoss() 
        label = label.long()
        intensity = intensity.unsqueeze(1)
        intensity = intensity.float()
        loss0=self._dice_loss(preds[0],mask)
        #loss0 = F.binary_cross_entropy_with_logits(preds[0], mask, reduction='mean')
        loss1 = crossEntropy(preds[1],label)
        loss2 = binaryCrossEntropy(preds[2],intensity) 

        return torch.stack([loss0,loss1,loss2])
