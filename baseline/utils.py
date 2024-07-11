
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, roc_curve, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
import numpy as np

class AverageMeter(object):
    def __init__(self, name=None, writer=None):
        self._writer = writer
        self._name = name
        self.reset()

    def reset(self):
        self.val = [0]
        self.avg = [0]
        self.sum = [0]
        self.count = 0

    def update(self, val, n=1):
        self.count += n
        if isinstance(val, list):
            # Extend the lists if they are shorter than the incoming list
            length_difference = len(val) - len(self.sum)
            if length_difference > 0:
                self.sum.extend([0] * length_difference)
                self.avg.extend([0] * length_difference)
                self.val.extend([0] * length_difference)

            for i, v in enumerate(val):
                self.sum[i] += v
                self.avg[i] = self.sum[i] / self.count
            self.val = val
        else:
            if not isinstance(self.sum, list):
                self.sum = [self.sum]
                self.avg = [self.avg]
                self.val = [self.val]
            self.sum[0] += val
            self.avg[0] = self.sum[0] / self.count
            self.val[0] = val

    def write(self, epoch=0):
        if self.count != 0:
            if isinstance(self.avg, list):
                for i, val in enumerate(self.avg):
                    try:
                        self._writer.add_scalar(f'{self._name}_{i}', val, epoch)
                    except Exception as e:
                        print(e)
            else:
                try:
                    self._writer.add_scalar(self._name, self.avg, epoch)
                except Exception as e:
                    print(e)

class LossMeter(object):
    def __init__(self, name=None, writer=None):
        self.reset()
        self._writer = writer
        self._name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, list):
            val = sum(val) / len(val)
            # log each value in the list
        self.val = val
        self.sum += val 
        self.count += n
        self.avg = self.sum / self.count

    def write(self, epoch=0):
        #  if self.avg is list then log each value
        if self.count !=0:
            if isinstance(self.avg, list):
                for i, val in enumerate(self.avg):
                    try:
                        self._writer.add_scalar(f'{self._name}_{i}', val, epoch)
                    except Exception as e:
                        print(e)
            else:
                try:
                    self._writer.add_scalar(self._name, self.avg, epoch)
                except Exception as e:
                    print(e)

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # y_pred = F.log_softmax(y_pred, dim=1)
        loss = F.cross_entropy(y_pred, y_true)
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
    def __init__(self, alpha=0.55, gamma=2.0, reduction='mean'):
        """
        Initializes the Focal Loss.

        Parameters:
        - alpha (float): Weighting factor for the rare class.
        - gamma (float): Modulating factor to focus on hard examples.
        - reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

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
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initializes the Focal Loss for multi-class classification.

        Parameters:
        - alpha (tensor): Weighting factor for each class, shape (num_classes,).
        - gamma (float): Modulating factor to focus on hard examples.
        - reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLossMultiClass, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the focal loss for multi-class classification.

        Parameters:
        - inputs (tensor): Predictions from the model, expected to be logits with shape (batch_size, num_classes).
        - targets (tensor): Ground truth labels, with shape (batch_size,) where each value is 0 <= targets[i] <= num_classes-1.
        """
        # Convert inputs to probabilities
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=probs.shape[1]).float()

        # Calculate the cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.sum(targets_one_hot * probs, dim=1)

        # Calculate the focal loss
        alpha_factor = self.alpha[targets] if self.alpha is not None else 1.0
        focal_loss = alpha_factor * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
        
class CCCLoss(nn.Module):
    def __init__(self):
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


def weighted_accuracy(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=1)
    return balanced_accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())

def EER(y_pred, y_true):

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    n_classes = y_pred.shape[1]
    y_true_binarized = label_binarize(y_true, classes=range(n_classes))
    eers = []

    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(y_true_binarized[:, i], y_pred[:, i])
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        eers.append(eer)

    average_eer = np.mean(eers)
    return average_eer

def f1_score_(predictions, labels):
    predictions = torch.argmax(predictions, dim=1)
    f1 = f1_score(predictions.cpu().numpy(), labels.cpu().numpy(), average='macro')
    return f1

def f1_score_per_class(predictions, labels):
    num_classes = predictions.shape[1]
    # apply sigmoid to the predictions and set 1 if greater than 0.5
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()
    f1 = []
    for i in range(num_classes):
        pred = predictions[:, i]
        label = labels[:, i]
        f1.append(f1_score(pred.cpu().numpy(), label.cpu().numpy()))    
    return f1