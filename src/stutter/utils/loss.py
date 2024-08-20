import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        # y_pred = torch.argmax(y_pred, dim=1)
        loss = F.cross_entropy(y_pred, y_true)
        return loss


class YOHOLoss(nn.Module):
    def __init__(self, **kwargs):
        super(YOHOLoss, self).__init__()
    def forward(self, y_pred, y_true):
        ind = np.max(y_true[:,2:],axis=1) 
        y_pred[:,0:2]  = y_pred[:,0:2] * np.repeat(ind[:,np.newaxis], repeats=2, axis=1)
        loss = F.mse_loss(y_pred, y_true)
        return loss
    

class CrossEntropyLossWithReg(nn.Module):
    def __init__(self, model, reg_coeff=0.01):
        super(CrossEntropyLossWithReg, self).__init__()
        self.model = model
        self.reg_coeff = reg_coeff

    def forward(self, y_pred, y_true):
        # Calculate the standard cross-entropy loss
        loss = F.cross_entropy(y_pred, y_true)
        
        # Calculate L2 regularization (weight decay)
        l2_reg = torch.tensor(0.).to(y_pred.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param)**2
        
        # Add the regularization term to the loss
        loss += self.reg_coeff * l2_reg
        return loss

class FocalLoss(nn.Module):
    __acceptable_params = ['weights', 'gamma', 'reduction']
    def __init__(self, **kwargs):
        """
        Initializes the Focal Loss.

        Parameters:
        - alpha (float): Weighting factor for the rare class.
        - gamma (float): Modulating factor to focus on hard examples.
        - reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        [setattr(self, k, v) for k, v in kwargs.items() if k in self.__acceptable_params]


    def forward(self, inputs, targets):
        """
        Compute the focal loss.

        Parameters:
        - inputs (tensor): Predictions from the model, expected to be logits.
        - targets (tensor): Ground truth labels, with the same shape as inputs.
        """
        # apply softmax to the inputs
        inputs = F.softmax(inputs, dim=1)
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
class FocalLossMultiClass(nn.Module):
    __acceptable_params = ['alpha', 'gamma', 'reduction', 'device']
    def __init__(self, **kwargs):
        super(FocalLossMultiClass, self).__init__()
        [setattr(self, k, v) for k, v in kwargs.items() if k in self.__acceptable_params]
        self.gamma = kwargs.get('gamma', 2)
        self.alpha = torch.tensor(self.alpha).to(self.device) if self.alpha is not None else None

    def forward(self, inputs, targets):
        
        # Convert inputs to probabilities
        probs = F.sigmoid(inputs)

        # Calculate the cross entropy loss
        bce = F.binary_cross_entropy(probs, targets, reduction='none')

        pt = torch.where(targets == 1, probs, 1 - probs)
        f_loss = (1 - pt) ** self.gamma * bce

        if self.alpha is not None:
            if self.alpha.dim() == 0:
                alpha = self.alpha.expand_as(targets)
            else:
                alpha = self.alpha
            alpha_t = torch.where(targets == 1, alpha, 1-alpha)
            f_loss = alpha_t * f_loss

        if self.reduction == 'mean':
            return f_loss.mean()
        elif self.reduction == 'sum':
            return f_loss.sum()
        else:
            return f_loss
        
class CCCLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CCCLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_true_mean = torch.mean(y_true, dim=1, keepdim=True)
        y_pred_mean = torch.mean(y_pred, dim=1, keepdim=True)
        
        y_true_var = torch.var(y_true, dim=1, unbiased=False)
        y_pred_var = torch.var(y_pred, dim=1, unbiased=False)
        
        covariance = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean), dim=1)
        
        ccc = (2 * covariance) / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean).squeeze() ** 2)
        
        ccc_loss = 1 - ccc.mean()

        return ccc_loss

class BCELoss(nn.Module):
    __acceptable_params = ['weights', 'device']
    def __init__(self, **kwargs):
        super(BCELoss, self).__init__()
        [setattr(self, k, v) for k, v in kwargs.items() if k in self.__acceptable_params]
        self.weights = torch.tensor(self.weights).to(self.device)

    def forward(self, y_pred, y_true):
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=self.weights)
        return loss

class BCELossWithLogits(nn.BCEWithLogitsLoss):
    def __init__(self, **kwargs):
        super(BCELossWithLogits, self).__init__(weight=torch.tensor(kwargs.get('weights', None)))
        print(f'Using BCELossWithLogits with weights: {kwargs.get("weights", None)}')
        
    def forward(self, y_pred, y_true):
        return super().forward(y_pred, y_true)
    
loss_registery = {
    'ccc': CCCLoss,
    'ce': CrossEntropyLoss,
    'focal': FocalLossMultiClass,
    'bce': BCELossWithLogits
}

def build_loss(cfg):
    criterion = {}
    for key, loss in zip(cfg.tasks, cfg.solver.losses):
        criterion[key] = loss_registery[loss](device=cfg.solver.device, **cfg.loss ).to(cfg.solver.device)
    return criterion