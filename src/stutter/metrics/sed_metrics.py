import torch
import numpy as np
from sklearn.metrics import f1_score

def compute_metrics(predictions, targets, num_classes, segment_length=1.0, event_iou_threshold=0.5):
    """
    Compute segment-based and event-based F1 scores per class for Sound Event Detection.

    Args:
        predictions (torch.Tensor): Model outputs of shape (B, T, C), where B is batch size, T is the temporal dimension,
                                    and C is the number of classes.
        targets (torch.Tensor): Ground truth of shape (B, T, C) in the same format as predictions.
        num_classes (int): Number of classes (C).
        segment_length (float): Length of each segment in seconds for segment-based evaluation.
        event_iou_threshold (float): IoU threshold for event-based evaluation.

    Returns:
        dict: Dictionary containing segment-based and event-based F1 scores per class.
    """
    # Convert predictions to binary values using a threshold (e.g., 0.5)
    predictions = (predictions > 0.5).int()
    targets = targets.int()

    # Initialize lists to collect per-class segment and event metrics
    segment_f1_scores_per_class = [[] for _ in range(num_classes)]
    event_f1_scores_per_class = [[] for _ in range(num_classes)]

    # Iterate over each batch item
    for i in range(predictions.size(0)):  # Loop over batches
        pred = predictions[i].cpu().numpy()  # (T, C)
        truth = targets[i].cpu().numpy()  # (T, C)

        # Compute segment-based F1 scores per class
        segment_f1_per_class = segment_based_f1_per_class(pred, truth, segment_length, num_classes)
        for c in range(num_classes):
            segment_f1_scores_per_class[c].append(segment_f1_per_class[c])

        # Compute event-based F1 scores per class
        event_f1_per_class = event_based_f1_per_class(pred, truth, event_iou_threshold, num_classes)
        for c in range(num_classes):
            event_f1_scores_per_class[c].append(event_f1_per_class[c])

    # Calculate the average F1 score per class across the batch
    avg_segment_f1_per_class = [np.mean(scores) if scores else 0 for scores in segment_f1_scores_per_class]
    avg_event_f1_per_class = [np.mean(scores) if scores else 0 for scores in event_f1_scores_per_class]

    return {
        "segment_f1_per_class": avg_segment_f1_per_class,
        "event_f1_per_class": avg_event_f1_per_class
    }

def segment_based_f1_per_class(pred, truth, segment_length, num_classes, sample_rate, hop_length):
    """
    Compute the segment-based F1 score for each class by dividing the prediction and truth into segments.

    Args:
        pred (np.ndarray): Predicted binary array of shape (T, C).
        truth (np.ndarray): Ground truth binary array of shape (T, C).
        segment_length (float): Segment length in seconds.
        num_classes (int): Number of classes.
        sample_rate (int): Sample rate of the audio.
        hop_length (int): Hop length used during feature extraction.

    Returns:
        list: Segment-based F1 scores per class.
    """
    # Calculate the number of frames per segment
    frames_per_segment = int(segment_length * sample_rate / hop_length)

    f1_scores = []

    # Iterate over each class to compute per-class F1 scores
    for c in range(num_classes):
        class_pred = pred[:, c]
        class_truth = truth[:, c]

        segment_f1_scores = []

        # Segment-wise F1 calculation
        for start in range(0, len(class_pred), frames_per_segment):
            end = start + frames_per_segment
            if end > len(class_pred):
                break

            pred_segment = class_pred[start:end].flatten()
            truth_segment = class_truth[start:end].flatten()

            f1 = f1_score(truth_segment, pred_segment, average='binary', zero_division=1)
            segment_f1_scores.append(f1)

        # Calculate the average F1 score for the class
        f1_scores.append(np.mean(segment_f1_scores) if segment_f1_scores else 0)

    return f1_scores

def event_based_f1_per_class(pred, truth, iou_threshold, num_classes):
    """
    Compute the event-based F1 score per class using Intersection over Union (IoU).

    Args:
        pred (np.ndarray): Predicted binary array of shape (T, C).
        truth (np.ndarray): Ground truth binary array of shape (T, C).
        iou_threshold (float): IoU threshold for event matching.
        num_classes (int): Number of classes.

    Returns:
        list: Event-based F1 scores per class.
    """
    f1_scores = []
    for c in range(num_classes):
        true_events = get_events(truth, c)
        pred_events = get_events(pred, c)
        print(f'Class {c}: {len(true_events)} true events, {len(pred_events)} predicted events')
        tp, fp, fn = 0, 0, 0
        for true_event in true_events:
            matched = False
            for pred_event in pred_events:
                iou = calculate_iou(true_event, pred_event)
                if iou >= iou_threshold:
                    tp += 1
                    pred_events.remove(pred_event)  # Remove matched prediction
                    matched = True
                    break
            if not matched:
                fn += 1

        fp = len(pred_events)  # Remaining unmatched predictions are false positives

        # Calculate F1 score
        if tp + fp + fn == 0:
            # f1_scores.append(1.0)  # Perfect score if no events are found at all
            pass
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            event_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(event_f1)
    return f1_scores

def get_events(binary_array, class_idx):
    """
    Extract events for a specific class from a binary array (onset-offset pairs).

    Args:
        binary_array (np.ndarray): Binary array of shape (T, C).
        class_idx (int): Index of the class to extract events for.

    Returns:
        list: List of detected events as (onset, offset).
    """
    events = []
    in_event = False
    onset = 0
    for t in range(binary_array.shape[0]):
        if binary_array[t, class_idx] == 1 and not in_event:
            onset = t
            in_event = True
        elif binary_array[t, class_idx] == 0 and in_event:
            events.append((onset, t - 1))
            in_event = False
    if in_event:
        events.append((onset, binary_array.shape[0] - 1))
    return events

def calculate_iou(event1, event2):
    """
    Calculate Intersection over Union (IoU) for two events.

    Args:
        event1 (tuple): First event as (onset, offset).
        event2 (tuple): Second event as (onset, offset).

    Returns:
        float: IoU value.
    """
    onset1, offset1 = event1
    onset2, offset2 = event2

    intersection = max(0, min(offset1, offset2) - max(onset1, onset2) + 1)
    union = max(offset1, offset2) - min(onset1, onset2) + 1
    iou = intersection / union

    return iou

class SegmentBasedMetric(object):
    __acceptable_params = ['segment_length', 'sample_rate', 'hop_length']
    def __init__(self, **kwargs):
        [setattr(self, k, v) for k, v in kwargs.items() if k in self.__acceptable_params]

    def __call__(self, predictions, targets):
        num_classes = targets.size(2)
        segment_f1_scores = [[] for _ in range(num_classes)]
        predictions = (predictions > 0.5).int()
        targets = targets.int()
        for i in range(predictions.size(0)):  # Loop over batches
            pred = predictions[i].cpu().numpy()  # (T, C)
            truth = targets[i].cpu().numpy()  # (T, C)

            # Compute segment-based F1 scores per class
            segment_f1_per_class = segment_based_f1_per_class(
                pred, truth, self.segment_length, num_classes, self.sample_rate, self.hop_length
            )
            for c in range(num_classes):
                segment_f1_scores[c].append(segment_f1_per_class[c])

        # Average F1 scores across the temporal dimension (batch)
        avg_segment_f1_per_class = [np.mean(scores) if scores else 0 for scores in segment_f1_scores]
        return avg_segment_f1_per_class

class EventBasedMetric(object):
    __acceptable_params = ['event_iou_threshold']
    def __init__(self, **kwargs):
        [setattr(self, k, v) for k, v in kwargs.items() if k in self.__acceptable_params]

    def __call__(self, predictions, targets):
        num_classes = targets.size(2)
        event_f1_scores = [[] for _ in range(num_classes)]
        predictions = (predictions > 0.5).int()
        targets = targets.int()
        for i in range(predictions.size(0)):  # Loop over batches
            pred = predictions[i].cpu().numpy()  # (T, C)
            truth = targets[i].cpu().numpy()  # (T, C)

            # Compute event-based F1 scores per class
            event_f1_per_class = event_based_f1_per_class(pred, truth, self.event_iou_threshold, num_classes)

            for c in range(num_classes):
                event_f1_scores[c].append(event_f1_per_class[c])

        # Average F1 scores across the temporal dimension (batch)
        avg_event_f1_per_class = [np.mean(scores) if scores else 0 for scores in event_f1_scores]
        return avg_event_f1_per_class


if __name__ == "__main__":
    # Test the functions with dummy data
    predictions = torch.tensor([
        [[0.1, 0.9], [0.3, 0.7], [0.8, 0.2]],
        [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]]
    ])
    targets = torch.tensor([
        [[0, 1], [0, 1], [1, 0]],
        [[1, 0], [0, 1], [1, 0]]
    ])

    kwargs = {
        'num_classes': 2,
        'segment_length': 1.0,
        'event_iou_threshold': 0.5,
        'sample_rate': 1,
        'hop_length': 1
    }

    eb = EventBasedMetric(**kwargs)
    event_f1_scores = eb(predictions, targets)
    print(event_f1_scores)

    sb = SegmentBasedMetric(**kwargs)
    segment_f1_scores = sb(predictions, targets)
    print(segment_f1_scores)

    # Expected output: {'segment_f1_per_class': [0.5, 0.5], 'event_f1_per_class': [0.5, 0.5]}
    # The F1 scores are 0.5 for both classes in both segment-based and event-based evaluation.