
import torch
import torch.nn as nn
import torch.nn.functional as F

class AverageMeter(object):
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
        self.val = val
        self.sum += val 
        self.count += n
        self.avg = self.sum / self.count

    def write(self, epoch=0):
        if self.count !=0:
            try:
                self._writer.add_scalar(self._name, self.avg, epoch)
            except Exception as e:
                print(e)

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = F.log_softmax(y_pred, dim=1)
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
    
    num_classes = y_pred.shape[1]

    y_pred = torch.argmax(y_pred, dim=1)

    # Calculate the accuracy for each class
    correct = (y_pred == y_true).float()
    class_counts = torch.bincount(y_true, minlength=num_classes)
    class_accuracies = torch.bincount(y_true, weights=correct, minlength=num_classes) / class_counts
    
    # Calculate the weighted accuracy
    weighted_accuracy = (class_accuracies * class_counts).sum() / class_counts.sum()

    return weighted_accuracy.item()

def EER(y_pred, y_true):

    y_pred = torch.argmax(y_pred, dim=1)
    # Calculate the false positive rate and true positive rate for each threshold
    thresholds = torch.linspace(0, 1, 1000)
    fpr = torch.tensor([((y_pred >= t) & (y_true == 0)).float().mean().item() for t in thresholds])
    tpr = torch.tensor([((y_pred >= t) & (y_true == 1)).float().mean().item() for t in thresholds])

    # Find the threshold that minimizes the difference between FPR and TPR
    diff = torch.abs(fpr - (1 - tpr))
    EER = fpr[diff.argmin()].item()

    return EER

def multi_class_EER(y_pred, y_true):
    EERs = []
    num_classes = y_pred.shape[1]
    for c in range(num_classes):
        # Consider current class as positive and all others as negative
        y_true_binary = (y_true == c).int()
        y_pred_binary = torch.softmax(y_pred, dim=1)[:, c]

        # Calculate the false positive rate and true positive rate for each threshold
        thresholds = torch.linspace(0, 1, 1000)
        fpr = torch.tensor([((y_pred_binary >= t) & (y_true_binary == 0)).float().mean().item() for t in thresholds])
        tpr = torch.tensor([((y_pred_binary >= t) & (y_true_binary == 1)).float().mean().item() for t in thresholds])

        # Find the threshold that minimizes the difference between FPR and TPR
        diff = torch.abs(fpr - (1 - tpr))
        EER = fpr[diff.argmin()].item()
        EERs.append(EER)

    # Calculate average EER across all classes
    avg_EER = sum(EERs) / num_classes
    return avg_EER

def f1_score(predictions, labels):

    predictions = torch.argmax(predictions, dim=1)

    true_positives = (predictions * labels).sum().item()
    predicted_positives = predictions.sum().item()
    actual_positives = labels.sum().item()

    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (actual_positives + 1e-8)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return f1

def f1_score_per_class(predictions, labels):
    num_classes = predictions.shape[1]

    predictions = torch.argmax(predictions, dim=1)
    # Convert predictions and labels to one-hot encoding if they aren't already
    predictions_one_hot = torch.eye(num_classes)[predictions]
    labels_one_hot = torch.eye(num_classes)[labels]

    true_positives = (predictions_one_hot * labels_one_hot).sum(dim=0)
    predicted_positives = predictions_one_hot.sum(dim=0)
    actual_positives = labels_one_hot.sum(dim=0)

    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (actual_positives + 1e-8)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return f1