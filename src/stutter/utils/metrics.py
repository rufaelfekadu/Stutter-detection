import torch
import numpy as np
from datasets import load_metric
from sklearn.metrics import f1_score, roc_curve, multilabel_confusion_matrix


acc = load_metric("accuracy")
f1 = load_metric("f1")
def compute_video_classification_metrics(p):
    accuracy  = acc.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)['accuracy']
    f1_score = f1.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)['f1']
    return {"accuracy": accuracy, "f1": f1_score}

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
    # predictions = torch.sigmoid(predictions)
    # predictions = (predictions > 0.5).float()
    f1 = f1_score(predictions.cpu().numpy(), labels.cpu().numpy(), average='macro')
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
        f1.append(f1_score(pred.cpu().numpy(), label.cpu().numpy(), average='macro'))    
    # append any class
    predictions_any =  (torch.sum(predictions[:, :num_classes-1], dim=1) >0 ).int()
    labels_any = (torch.sum(labels[:, :num_classes-1], dim=1) >0).int()
    f1.append(f1_score(predictions_any.cpu().numpy(), labels_any.cpu().numpy(), average='macro'))
    return f1
