import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # y_pred = F.softmax(y_pred, dim=1)
        # y_pred = torch.argmax(y_pred, dim=1).float()
        loss = F.cross_entropy(y_pred, y_true.long(), weight=torch.tensor(0.9).to(y_true.device))
        return loss

class FocalLoss(nn.Module):
    __acceptable_params = ['alpha', 'gamma', 'reduction', 'device']
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
        self.alpha = torch.tensor(self.alpha[0]).to(self.device)


    def forward(self, inputs, targets):
        """
        Compute the focal loss.

        Parameters:
        - inputs (tensor): Predictions from the model, expected to be logits.
        - targets (tensor): Ground truth labels, with the same shape as inputs.
        """
        # apply softmax to the inputs
        inputs = F.sigmoid(inputs)
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
    __acceptable_params = ['weights','alpha', 'gamma', 'reduction', 'device']
    def __init__(self, **kwargs):
        super(FocalLossMultiClass, self).__init__()
        [setattr(self, k, v) for k, v in kwargs.items() if k in self.__acceptable_params]
        self.gamma = kwargs.get('gamma', 2)
        self.alpha = torch.tensor(self.alpha).to(self.device) if self.alpha is not None else None
        self.weights = torch.tensor(self.weights).to(self.device) 

    def forward(self, inputs, targets):
        
        # Convert inputs to probabilities
        probs = F.sigmoid(inputs)
        # Calculate the cross entropy loss
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none',pos_weight=self.weights.to(targets.device))

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

class WeightedMAELoss(nn.Module):
    __acceptable_params = ['weights']
    def __init__(self, **kwargs):
        super(WeightedMAELoss, self).__init__()
        [setattr(self, k, v) for k, v in kwargs.items() if k in self.__acceptable_params]
        # self.class_weights = class_weights
        self.weights =torch.tensor([0.5, 0.2, 0.5, 1.0], dtype=torch.float32)
 
    def forward(self, outputs, targets):
        loss = 0.0
        for i in range(len(self.weights)):
            weight = self.weights[i]
            if (targets == i).sum() > 0:
                loss += weight * torch.abs(outputs[targets == i] - targets[targets == i]).mean()
        return loss
    
class BCELoss(nn.Module):
    __acceptable_params = ['weights', 'device']
    def __init__(self, **kwargs):
        super(BCELoss, self).__init__()
        [setattr(self, k, v) for k, v in kwargs.items() if k in self.__acceptable_params]

    def forward(self, y_pred, y_true):
        weights = torch.tensor([self.weights[0] if label==1 else 1-self.weights[0] for label in y_true]).to(self.device)
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=weights)
        return loss

class BCELossMulticlass(nn.Module):
    __acceptable_params = ['weights', 'device']
    def __init__(self, **kwargs):
        super(BCELossMulticlass, self).__init__()
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
    


