import torch
import numpy as np
from datasets import load_metric
from sklearn.metrics import f1_score, roc_curve, multilabel_confusion_matrix
import torcheval.metrics.functional as F


acc = load_metric("accuracy")
f1 = load_metric("f1")



def accuracy(pred, label):
    if pred.shape[1] ==1:
        pred = torch.sigmoid(pred)
        return F.multilabel_accuracy(pred, label, threshold=0.5)
    else:
        pred = torch.sigmoid(pred)
        return F.multilabel_accuracy(pred, label, threshold=0.5, criteria='hamming')

def f1_score_multilabel(pred, label):

    if len(pred.shape)==1:
        pred = torch.sigmoid(pred)
        return F.binary_f1_score(pred, label, threshold=0.5)
    else:
        f1 = []
        for i in range(pred.shape[1]):
            pred[:, i] = torch.sigmoid(pred[:, i])
            f1.append(F.binary_f1_score(pred[:, i], label[:, i], threshold=0.5))
        return f1

def binary_eer(pred, label):
    pred = torch.sigmoid(pred)
    cm = F.binary_confusion_matrix(pred, label, threshold=0.5)
    # Extract true positives, false positives, false negatives, and true negatives
    tp = cm[1, 1].item()
    fp = cm[0, 1].item()
    fn = cm[1, 0].item()
    tn = cm[0, 0].item()
    
    # Calculate False Acceptance Rate (FAR) and False Rejection Rate (FRR)
    FAR = fp / (fp + tn)
    FRR = fn / (fn + tp)
    
    # Calculate EER
    EER = (FAR + FRR) / 2
    return EER

def eer_multilabel(pred, label):

    if pred.shape[1] ==1:
        pred = torch.sigmoid(pred)
        return binary_eer(pred, label)

    else:
        eers = []
        for i in range(pred.shape[1]):
            pred[:, i] = torch.sigmoid(pred[:, i])
            eers.append(binary_eer(pred[:, i], label[:, i]))
        return eers

def compute_video_classification_metrics(p):
    accuracy  = acc.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)['accuracy']
    f1_score = f1.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)['f1']
    return {"accuracy": accuracy, "f1": f1_score}

def binary_acc(y_true, y_pred):
  
    threshold = 0.5

    binary_true = y_true[:,:,2:-1]
    binary_pred = y_pred[:,:,2:-1]
    binary_pred = torch.sigmoid(binary_pred) >= threshold
    # binary_true = torch.greater_equal(binary_true, threshold)
    # binary_pred = torch.greater_equal(binary_pred, threshold)
    # binary_pred = tf.greater(y_pred[:, [0, 2]], threshold)

    # acc = tf.square(y_true - y_pred)

    acc = torch.sum(binary_true == binary_pred).float() / binary_true.numel()

    return acc

def iou_metric(y_true, y_pred):
    # y_true -> (batch_size, 22, 14)
    # y_pred -> (batch_size, 22, 14)
    threshold = 0.5
    # for all dim=1 if any class prediction is greater than threshold then set it to 1
    y_pred_b = torch.sigmoid(y_pred[:,:,2:-1]) >= threshold
    y_true_b = y_true[:,:,2:-1]

    mask = y_true_b.sum(dim=2) > 0
    # compute iou between start and end times in seconds
    start_pred, end_pred = y_pred[:,:,0], y_pred[:,:,1]
    start_true, end_true = y_true[:,:,0], y_true[:,:,1]

    intersection_start = torch.max(start_pred, start_true)
    intersection_end = torch.min(end_pred, end_true)
    intersection = torch.clamp(intersection_end - intersection_start, min=0)

    union_start = torch.min(start_pred, start_true)
    union_end = torch.max(end_pred, end_true)
    union = torch.clamp(union_end - union_start, min=0)

    iou = intersection / (union + 1e-6)

    iou = iou * mask.float()
    iou = iou.sum() / mask.sum()

    return iou.mean()

def binary_f1(y_true, y_pred):
    threshold = 0.5
    binary_true = y_true[:,:,2:-1]
    binary_pred = y_pred[:,:,2:-1]
    binary_pred = torch.sigmoid(binary_pred) >= threshold
    num_classes = binary_true.shape[-1]
    f1 = []
    for i in range(num_classes):
        pred = binary_pred[:,:, i]
        label = binary_true[:,:, i]
        f1.append(f1_score(pred.cpu().numpy(), label.cpu().numpy(), average='macro'))
    return f1

def EER(predictions, labels):

    # if  isinstance(predictions, torch.Tensor):
    #     predictions = (torch.sigmoid(predictions)>0.5).float()
    #     predictions, labels = predictions.cpu().numpy(), labels.cpu().numpy()

    fpr, tpr, thresholds = roc_curve(labels, predictions)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    return eer, eer_threshold

def multilabel_EER(predictions, labels):

    # predictions = (torch.sigmoid(predictions) > 0.5).float()
    predictions, labels = predictions.cpu().numpy(), labels.cpu().numpy()

    n_labels = predictions.shape[1]
    eers = []
    
    for i in range(n_labels):
        eer, eer_threshold = EER(labels[:, i], predictions[:, i])
        eers.append(eer)
    
    overall_eer = np.mean(eers)
    return overall_eer

def weighted_accuracy(predictions, labels):

    # predictions = (torch.sigmoid(predictions) > 0.5).float()
    predictions, labels = predictions.cpu().numpy(), labels.cpu().numpy()

    cm = multilabel_confusion_matrix(labels, predictions)
    
    accuracies = []
    supports = []
    
    for i in range(cm.shape[0]):
        tn, fp, fn, tp = cm[i].ravel()
        
        # Accuracy for label i
        accuracy_i = (tp + tn) / (tp + tn + fp + fn)
        accuracies.append(accuracy_i)
        
        # Support for label i
        support_i = tp + fn
        supports.append(support_i)
   
    # Compute weighted accuracy
    weighted_accuracy = np.average(accuracies, weights=supports)
    
    return weighted_accuracy

def f1_score_(predictions, labels):
    # predictions = torch.argmax(predictions, dim=1)
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()
    f1 = f1_score(predictions.cpu().numpy(), labels.cpu().numpy(), average='weighted')
    return f1

def f1_score_per_class(predictions, labels):
    num_classes = predictions.shape[1]
    ## apply sigmoid to the predictions and set 1 if greater than 0.5
    # predictions = torch.sigmoid(predictions)
    # predictions = (predictions > 0.5).float()
    f1 = []
    for i in range(num_classes):
        pred = predictions[:, i]
        label = labels[:, i]
        f1.append(f1_score(pred.cpu().numpy(), label.cpu().numpy()))    
    # append any class
    predictions_any =  (torch.sum(predictions, dim=1) >0 ).int()
    labels_any = (torch.sum(labels, dim=1) >0).int()
    f1.append(f1_score(predictions_any.cpu().numpy(), labels_any.cpu().numpy()))
    return f1


metric_registery={
    'accuracy': accuracy,
    'f1': f1_score_multilabel,
    'eer': eer_multilabel,
}