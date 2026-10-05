"""Time-Aware Temporal Transformer for Longitudinal Diagnostic Delay Prediction.

Integrates continuous time-gap encodings (Time2Vec / exponential temporal encodings)
with Transformer Multi-Head Self-Attention for longitudinal trajectory modeling.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Time2Vec(nn.Module):
    """Continuous Time-to-Vector encoding layer.
    
    Transforms scalar time intervals into periodic and linear representation vectors.
    """

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        # Linear component
        self.w0 = nn.Parameter(torch.randn(1, 1))
        self.b0 = nn.Parameter(torch.randn(1, 1))
        # Periodic components
        self.w = nn.Parameter(torch.randn(1, out_dim - 1))
        self.b = nn.Parameter(torch.randn(1, out_dim - 1))

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_t: (Batch, Seq_Len, 1) or (Batch, Seq_Len)
            
        Returns:
            time_features: (Batch, Seq_Len, out_dim)
        """
        if delta_t.dim() == 2:
            delta_t = delta_t.unsqueeze(-1)
        # Linear projection
        v0 = delta_t * self.w0 + self.b0
        # Periodic projection
        vp = torch.sin(delta_t * self.w + self.b)
        return torch.cat([v0, vp], dim=-1)


class LongitudinalTemporalTransformer(nn.Module):
    """Time-Aware Transformer Encoder for patient post trajectories."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        time_dim: int = 16,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.d_model = d_model
        
        # Time gap encoding
        self.time2vec = Time2Vec(out_dim=time_dim)
        
        # Combined feature projection: text & lexicon features + time2vec features
        self.feature_proj = nn.Linear(input_dim + time_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        delta_t: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (Batch, Seq_Len, Input_Dim)
            delta_t: (Batch, Seq_Len, 1)
            mask: (Batch, Seq_Len) boolean mask (True = valid post, False = pad)
            
        Returns:
            logits: (Batch,)
        """
        time_emb = self.time2vec(delta_t)
        combined = torch.cat([x, time_emb], dim=-1)
        h = self.layer_norm(self.feature_proj(combined))

        # PyTorch Transformer padding mask: True indicates ignored padding position
        src_key_padding_mask = ~mask if mask is not None else None

        encoded = self.transformer_encoder(h, src_key_padding_mask=src_key_padding_mask)

        # Global average pooling over non-padded positions
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            sum_encoded = torch.sum(encoded * mask_expanded, dim=1)
            count = torch.clamp(mask_expanded.sum(dim=1), min=1.0)
            pooled = sum_encoded / count
        else:
            pooled = encoded.mean(dim=1)

        logits = self.classifier(pooled).squeeze(-1)
        return logits
