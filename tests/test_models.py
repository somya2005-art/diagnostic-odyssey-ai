"""Unit tests for models and neural sequence architectures."""

import unittest
import numpy as np
import torch
from src.models.baselines import BaselineModelSuite
from src.models.bilstm_attention import LongitudinalBiLSTMAttention, TimelineAttention
from src.models.temporal_transformer import LongitudinalTemporalTransformer, Time2Vec


class TestModelArchitectures(unittest.TestCase):

    def test_baselines_cv(self):
        np.random.seed(42)
        X = np.random.randn(40, 10)
        y = np.random.randint(0, 2, size=40)
        
        suite = BaselineModelSuite(random_state=42)
        results_df = suite.fit_and_evaluate_cv(X, y, n_splits=3)
        
        self.assertGreater(len(results_df), 0)
        self.assertIn("F1_Score", results_df.columns)
        self.assertIn("ROC_AUC", results_df.columns)

    def test_bilstm_attention_shapes(self):
        batch_size = 4
        seq_len = 10
        input_dim = 32
        hidden_dim = 16

        model = LongitudinalBiLSTMAttention(input_dim=input_dim, hidden_dim=hidden_dim)
        x = torch.randn(batch_size, seq_len, input_dim)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[:, 7:] = False  # last 3 positions padded

        logits, attn_weights = model(x, mask=mask)

        self.assertEqual(logits.shape, (batch_size,))
        self.assertEqual(attn_weights.shape, (batch_size, seq_len))
        
        # Check attention weights sum to 1.0 (approximately)
        attn_sums = attn_weights.sum(dim=-1).detach().numpy()
        np.testing.assert_allclose(attn_sums, np.ones(batch_size), atol=1e-4)

    def test_temporal_transformer_shapes(self):
        batch_size = 4
        seq_len = 8
        input_dim = 24

        model = LongitudinalTemporalTransformer(input_dim=input_dim, d_model=32, time_dim=8, nhead=2)
        x = torch.randn(batch_size, seq_len, input_dim)
        delta_t = torch.rand(batch_size, seq_len, 1) * 30.0
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        logits = model(x, delta_t, mask=mask)
        self.assertEqual(logits.shape, (batch_size,))


if __name__ == "__main__":
    unittest.main()
