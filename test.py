def weighted_accuracy(true_labels, predicted_labels):
    from sklearn.metrics import multilabel_confusion_matrix
    import numpy as np
    
    # Compute confusion matrix for each label
    cm = multilabel_confusion_matrix(true_labels, predicted_labels)
    
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


if __name__ == '__main__':
    # Example usage
    true_labels = [[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]]
    predicted_labels = [[1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 0, 0]]

    print(weighted_accuracy(true_labels, predicted_labels))