import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithLabelSmoothing(nn.Module):
    __acceptable_params = ['alpha', 'gamma', 'smoothing']
    def __init__(self, **kwargs):
        """
        Focal Loss with Label Smoothing for Sound Event Detection.

        Args:
            num_classes (int): Number of classes (C).
            alpha (float): Balance factor for rare classes; higher value puts more emphasis on rare classes.
            gamma (float): Focusing parameter; larger values focus more on hard-to-classify examples.
            smoothing (float): Label smoothing factor; reduces confidence of the model by softening labels.
        """
        super(FocalLossWithLabelSmoothing, self).__init__()
        [setattr(self, k, v) for k, v in kwargs.items() if k in self.__acceptable_params]
        self.smoothing = kwargs.get('smoothing', 0.1)  # Default smoothing value

    def forward(self, inputs, targets):
        """
        Forward pass of the loss function.

        Args:
            inputs (torch.Tensor): Logits of shape (B, T, C).
            targets (torch.Tensor): Ground truth labels of shape (B, T, C).
        
        Returns:
            torch.Tensor: Calculated loss value.
        """
        # Apply sigmoid to logits to get probabilities
        inputs = inputs.sigmoid()
        
        # Apply label smoothing
        targets = targets * (1 - self.smoothing) + self.smoothing / targets.size(2)
        
        # Compute Binary Cross Entropy loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Compute the modulating factor (focal factor)
        pt = torch.where(targets == 1, inputs, 1 - inputs)  # Probability of the correct class
        focal_weight = self.alpha[0] * (1 - pt) ** self.gamma  # Focal weight
        
        # Apply focal factor to the BCE loss
        loss = focal_weight * bce_loss
        
        return loss.mean()
    
class SedLoss(nn.Module):
    __acceptable_params = ['weights', 'device', 'smoothing']
    
    def __init__(self, **kwargs):
        super(SedLoss, self).__init__()
        [setattr(self, k, kwargs.get(k, None)) for k in self.__acceptable_params]
        self.weights = torch.tensor(self.weights)
        self.smoothing = kwargs.get('smoothing', 0.1)  # Default smoothing value

    def forward(self, y_pred, y_true):
        # y_pred -> (batch_size, 1500, 5)
        # y_true -> (batch_size, 1500, 5)
        # self.weights -> (5,1)
        
        # Apply label smoothing
        y_true_smoothed = y_true * (1 - self.smoothing) + self.smoothing * 0.5

        # Repeat weights to match the shape of y_true
        # weight = self.weights.repeat(y_true.shape[0], y_true.shape[1], 1).to(self.device)
        weight = self.weights.to(y_true.device)

        # Apply weights to the loss in the class dimension
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true_smoothed, weight=weight)
        return loss

class YOHOLoss(nn.Module):
    def __init__(self, **kwargs):
        super(YOHOLoss, self).__init__()
    def forward(self, y_pred, y_true):
        ind = torch.max(y_true[:,:,2:],axis=2)[0] 
        y_pred[:,:,0:2]  = y_pred[:,:,0:2] * torch.repeat_interleave(ind[:,:,None], repeats=2, dim=2)
        y_pred[:,:,2:]  = torch.sigmoid(y_pred[:,:,2:])
        loss = F.mse_loss(y_pred, y_true)
        return loss

if __name__ == "__main__":
    y_pred = torch.rand(32, 1500, 6)
    y_true = torch.rand(32, 1500, 6)
    kwargs = {
        'alpha': 0.5,
        'gamma': 2,
        'smoothing': 0.1
    }
    focal_loss = FocalLossWithLabelSmoothing(**kwargs)
    loss = focal_loss(y_pred, y_true)
    print(loss)
