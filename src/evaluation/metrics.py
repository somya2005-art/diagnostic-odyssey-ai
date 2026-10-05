"""Comprehensive Evaluation Metrics for Diagnostic Delay Prediction.

Calculates Accuracy, F1, Precision, Recall, Specificity, ROC-AUC, PR-AUC,
Brier score, and 95% Bootstrap Confidence Intervals for publication rigor.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix
)


class DiagnosticEvaluationSuite:
    """Computes comprehensive evaluation metrics and statistical confidence intervals."""

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculates standard classification and discrimination metrics."""
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)
        y_prob = np.array(y_prob, dtype=float)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        brier = brier_score_loss(y_true, y_prob)

        return {
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "Specificity": round(float(specificity), 4),
            "F1_Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "ROC_AUC": round(float(roc_auc), 4),
            "PR_AUC": round(float(pr_auc), 4),
            "Brier_Score": round(float(brier), 4),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn)
        }

    @staticmethod
    def bootstrap_confidence_intervals(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bootstraps: int = 500,
        alpha: float = 0.05,
        seed: int = 42
    ) -> Dict[str, Tuple[float, float]]:
        """Computes 95% bootstrap confidence intervals for ROC-AUC and PR-AUC."""
        rng = np.random.RandomState(seed)
        roc_scores = []
        pr_scores = []
        n = len(y_true)

        for _ in range(n_bootstraps):
            indices = rng.randint(0, n, n)
            if len(np.unique(y_true[indices])) < 2:
                continue
            roc_scores.append(roc_auc_score(y_true[indices], y_prob[indices]))
            pr_scores.append(average_precision_score(y_true[indices], y_prob[indices]))

        roc_ci = (
            round(float(np.percentile(roc_scores, 100 * (alpha / 2))), 4),
            round(float(np.percentile(roc_scores, 100 * (1 - alpha / 2))), 4)
        )
        pr_ci = (
            round(float(np.percentile(pr_scores, 100 * (alpha / 2))), 4),
            round(float(np.percentile(pr_scores, 100 * (1 - alpha / 2))), 4)
        )

        return {
            "ROC_AUC_95CI": roc_ci,
            "PR_AUC_95CI": pr_ci
        }
