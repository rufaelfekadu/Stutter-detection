import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import CrossEntropyLossWithReg, FocalLoss, FocalLossMultiClass, CCCLoss, weighted_accuracy, EER, multi_class_EER, f1_score, f1_score_per_class

class TestLossesAndMetrics(unittest.TestCase):
    def setUp(self):
        self.model = nn.Linear(10, 5)  # Dummy model for testing
        # Specific dummy values for each test
        self.y_pred_classification = torch.tensor([
            [-2.0, -1.5, 1.0, 0.5, -0.5],
            [-1.0, -1.5, 0.5, 2.0, -0.5],
            [-1.5, -2.0, 1.5, 0.0, -0.5],
            [-2.5, -0.5, 1.0, 0.5, -1.5],
            [-1.0, -0.5, 1.5, 2.0, -1.5]
        ], dtype=torch.float32)  # Logits for classification
        self.y_true_classification = torch.tensor([2, 3, 2, 2, 1], dtype=torch.long)  # Ground truth labels for classification


    def test_CrossEntropyLossWithReg(self):
        criterion = CrossEntropyLossWithReg(self.model)
        loss = criterion(self.y_pred_classification, self.y_true_classification)
        self.assertTrue(torch.is_tensor(loss))
        # Add a specific value check if desired

    def test_FocalLoss(self):
        criterion = FocalLoss()
        loss = criterion(self.y_pred_classification, self.y_true_classification)
        self.assertTrue(torch.is_tensor(loss))
        # Add a specific value check if desired
        expected_loss = torch.tensor(0.2169)  # Example expected value; adjust based on actual calculation
        self.assertTrue(torch.allclose(loss, expected_loss, atol=1e-4))

    def test_FocalLossMultiClass(self):
        criterion = FocalLossMultiClass(alpha=torch.rand(5))
        loss = criterion(self.y_pred_classification, self.y_true_classification)
        self.assertTrue(torch.is_tensor(loss))
        # Add a specific value check if desired

    def test_CCCLoss(self):
        criterion = CCCLoss()
        loss = criterion(self.y_pred_regression, self.y_true_regression)
        self.assertTrue(torch.is_tensor(loss))
        # Add a specific value check if desired

    def test_weighted_accuracy(self):
        acc = weighted_accuracy(self.y_pred_classification, self.y_true_classification)
        self.assertIsInstance(acc, float)
        # Add a specific value check if desired

    def test_EER(self):
        eer = EER(self.y_pred_classification, self.y_true_classification)
        self.assertIsInstance(eer, float)
        # Add a specific value check if desired

    def test_multi_class_EER(self):
        eer = multi_class_EER(self.y_pred_classification, self.y_true_classification)
        self.assertIsInstance(eer, float)
        # Add a specific value check if desired

    def test_f1_score(self):
        score = f1_score(self.y_pred_classification, self.y_true_classification)
        self.assertIsInstance(score, float)
        # Add a specific value check if desired

    def test_f1_score_per_class(self):
        scores = f1_score_per_class(self.y_pred_classification, self.y_true_classification)
        self.assertTrue(torch.is_tensor(scores))
        self.assertEqual(scores.shape[0], 5)  # Assuming 5 classes from y_pred
        # Add a specific value check if desired

if __name__ == '__main__':
    unittest.main()