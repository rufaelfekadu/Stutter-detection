
import torch
import torch.nn as nn

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



class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, y_true, y_pred):
        y_true_mean = torch.mean(y_true, dim=1, keepdim=True)
        y_pred_mean = torch.mean(y_pred, dim=1, keepdim=True)
        
        y_true_var = torch.var(y_true, dim=1, unbiased=False)
        y_pred_var = torch.var(y_pred, dim=1, unbiased=False)
        
        covariance = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean), dim=1)
        
        ccc = (2 * covariance) / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean).squeeze() ** 2)
        
        ccc_loss = 1 - ccc.mean()

        return ccc_loss