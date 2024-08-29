import unittest
import torch
from stutter.utils.metrics import iou_metric  # Assuming iou_metric is defined in metrics.py

class TestIoUMetric(unittest.TestCase):
    def test_iou_metric(self):
        # Test case 1: Perfect overlap
        y_true = torch.tensor([[[0, 10, 1, 1], [20, 30, 1, 1]]])
        y_pred = torch.tensor([[[0, 10, 1, 1], [20, 30, 1, 1]]])
        expected_iou = 1.0
        self.assertAlmostEqual(iou_metric(y_true, y_pred).item(), expected_iou, places=4)

        # Test case 2: Partial overlap
        y_true = torch.tensor([[[0, 10, 1, 1], [20, 30, 1, 1]]])
        y_pred = torch.tensor([[[5, 15, 1, 1], [25, 35, 1, 1]]])
        expected_iou = 0.3333  # (5/15 + 5/15) / 2
        self.assertAlmostEqual(iou_metric(y_true, y_pred).item(), expected_iou, places=4)

        # Test case 3: No overlap
        y_true = torch.tensor([[[0, 10, 1, 1], [20, 30, 1, 1]]])
        y_pred = torch.tensor([[[10, 20, 1, 1], [30, 40, 1, 1]]])
        expected_iou = 0.0
        self.assertAlmostEqual(iou_metric(y_true, y_pred).item(), expected_iou, places=4)

        # Test case 4: Mixed overlap
        y_true = torch.tensor([[[0, 10, 0, 0], [20, 30, 1, 1]]])
        y_pred = torch.tensor([[[5, 15, 1, 1], [20, 30, 1, 1]]])
        expected_iou = 1  # (5/15 + 10/10) / 2
        self.assertAlmostEqual(iou_metric(y_true, y_pred).item(), expected_iou, places=4)

if __name__ == '__main__':
    unittest.main()