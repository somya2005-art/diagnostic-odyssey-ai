"""Longitudinal Bidirectional LSTM with Post-Level Attention for Diagnostic Delay Prediction.

Models patient narrative trajectories across time, learning post-level attention weights
that quantify the diagnostic delay predictive risk of each clinical encounter.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class TimelineAttention(nn.Module):
    """Bahdanau / Additive Attention mechanism over longitudinal post representations."""

    def __init__(self, hidden_dim: int, attn_dim: int = 64):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, attn_dim, bias=True)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        lstm_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_outputs: (Batch, Seq_Len, Hidden_Dim * 2)
            mask: (Batch, Seq_Len) boolean mask where True = valid post, False = padded
            
        Returns:
            context_vector: (Batch, Hidden_Dim * 2)
            attention_weights: (Batch, Seq_Len)
        """
        # Score calculation: (B, T, attn_dim)
        scores = torch.tanh(self.W_a(lstm_outputs))
        # Unnormalized attention logits: (B, T, 1) -> (B, T)
        attn_logits = self.v_a(scores).squeeze(-1)

        if mask is not None:
            # Mask out padded steps with very large negative number
            attn_logits = attn_logits.masked_fill(~mask, -1e9)

        # Normalized attention weights: (B, T)
        attention_weights = F.softmax(attn_logits, dim=-1)
        
        # Replace NaNs if all positions were masked
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        # Context vector as weighted sum: (B, 1, T) x (B, T, 2H) -> (B, 1, 2H) -> (B, 2H)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_outputs).squeeze(1)

        return context_vector, attention_weights


class LongitudinalBiLSTMAttention(nn.Module):
    """Full Longitudinal BiLSTM + Attention sequence classifier for diagnostic delay."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        attn_dim: int = 48
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Timeline Attention Layer (Hidden size is 2 * hidden_dim due to bidirectionality)
        self.attention = TimelineAttention(hidden_dim * 2, attn_dim=attn_dim)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (Batch, Seq_Len, Input_Dim)
            mask: (Batch, Seq_Len) boolean mask (True = valid post)
            
        Returns:
            logits: (Batch, 1)
            attention_weights: (Batch, Seq_Len)
        """
        proj = self.input_proj(x)
        lstm_out, _ = self.lstm(proj)
        context, attn_weights = self.attention(lstm_out, mask=mask)
        logits = self.classifier(context)
        return logits.squeeze(-1), attn_weights


class PatientTimelineDataset(Dataset):
    """PyTorch Dataset for longitudinal patient timelines."""

    def __init__(
        self,
        feature_matrices: List[np.ndarray],
        labels: List[int],
        masks: List[np.ndarray]
    ):
        self.X = [torch.tensor(m, dtype=torch.float32) for m in feature_matrices]
        self.y = torch.tensor(labels, dtype=torch.float32)
        self.masks = [torch.tensor(m, dtype=torch.bool) for m in masks]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.masks[idx]


class BiLSTMTrainer:
    """Trainer and evaluator for Longitudinal BiLSTM Attention model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = "cpu"
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = LongitudinalBiLSTMAttention(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch, mask_batch in dataloader:
            X_batch, y_batch, mask_batch = X_batch.to(self.device), y_batch.to(self.device), mask_batch.to(self.device)
            self.optimizer.zero_grad()
            logits, _ = self.model(X_batch, mask=mask_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(1, len(dataloader))

    def evaluate(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        all_probs, all_preds, all_targets = [], [], []
        with torch.no_grad():
            for X_batch, y_batch, mask_batch in dataloader:
                X_batch, mask_batch = X_batch.to(self.device), mask_batch.to(self.device)
                logits, _ = self.model(X_batch, mask=mask_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(y_batch.numpy())

        return np.array(all_probs), np.array(all_preds), np.array(all_targets)

    def get_attention_for_patient(self, X_patient: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Returns attention weights for a single patient trajectory."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_patient, dtype=torch.float32).unsqueeze(0).to(self.device)
            mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(self.device)
            _, attn_weights = self.model(X_tensor, mask=mask_tensor)
            return attn_weights.squeeze(0).cpu().numpy()
